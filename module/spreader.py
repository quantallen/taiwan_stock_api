import asyncio
import json
import time
import logging
import sys
import socket
import websockets
from module.BTSEREST import hex_sign_msg
from math import floor
from pricer import Pricer
from functools import partial

#from predictor import Predictor
from predictor_change_spreadstamp import Predictor
#from predictor_change_close import Predictor
from position import Positions
from datetime import timedelta, datetime
from log_format import SaveLog
from order_book import OrderBook
from decimal import Decimal
import asyncio, ssl, websockets
import traceback
from binance.streams import BinanceSocketManager
import telegram

#todo kluge
#HIGHLY INSECURE
# ssl_context = ssl.SSLContext()
# ssl_context.check_hostname = False
# ssl_context.verify_mode = ssl.CERT_NONE
import requests
from json import loads
'''
logger = logging.getLogger(__name__)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
'''

class Spreader:
    production_endpoint = f'wss://fstream.binance.com/ws'
    chat_id = '-642791530'
    bot = telegram.Bot(token=('5384131643:AAFd62LyZl5mfI-Tzd0c_xTUYRKcRWugWpc'))

    orderbook = {}
    orderbook_5min = {}
    trades = {}
    ob_ref = OrderBook(max_depth=10)
    ob_target = OrderBook(max_depth=10)
    Ref_SeqNum = 0
    Target_SeqNum = 0
    def __init__(self, api, api2, config):
        logging.getLogger('').handlers = []
        self.bm2 = BinanceSocketManager(api2)
        self.bm = BinanceSocketManager(api)
        self.config = config
        self.log = SaveLog("Allen","PairTrading","FutureBTC_AVAX","/home/btsemm/")
        #self.log = None
        self.predictor = Predictor(
            window_size=self.config.MA_WINDOW_SIZE,
            ref_symbol=config.REFERENCE_SYMBOL,
            target_symbol=config.TARGET_SYMBOL,
            slippage=config.SLIPPAGE,
            log = self.log,
        )
        self.pricer = Pricer(
            api,
            config.REFERENCE_SYMBOL,
            config.TARGET_SYMBOL,
            self.log,
            self.config
        )
        self.spread_prices = None
        self.remember_quotos = None 
        self.api = api
    # async def listen_ws(self,topics,endpoint) :
    #     websocket = await websockets.connect(endpoint,ssl=ssl_context)
    #     print(topics)
    #     await websocket.send(json.dumps({
    #     "method": "SUBSCRIBE",
    #     "params":
    #     [
    #     topics
    #     ],
    #     "id": 1
    #     }))
    #     #response = await websocket.recv()
    #     return websocket
    async def Update_orderbook(self,task_queue, symbol):
        ws = self.bm.futures_depth_socket(symbol)
        async with ws as wscm:
            while True:
                await asyncio.sleep(0.001)
                resp = await wscm.recv()
                if 'data' not in resp:
                    print(resp)
                    continue

                data = resp['data']
                #print(data)
                
                timestamp = datetime.fromtimestamp(int(round(datetime.now().timestamp())))
                if symbol not in self.orderbook_5min or timestamp - self.orderbook_5min[symbol]['timestamp'] >= timedelta(seconds=self.config.TEST_SECOND):
                    self.orderbook_5min[symbol] = {
                            'buyQuote': [{'price': data['b'][0][0], 'size':data['b'][0][1]}],
                                    'sellQuote': [{'price': data['a'][0][0], 'size':data['a'][0][1]}],
                                    'timestamp': timestamp}   
                    self.predictor.update_spreads(self.orderbook_5min)
                
                self.orderbook[symbol] = {
                                    'buyQuote': [{'price': [data['b'][0][0],data['b'][1][0],data['b'][2][0]], 'size':[data['b'][0][1],data['b'][1][1],data['b'][2][1]]}],
                                    'sellQuote': [{'price': [data['a'][0][0],data['a'][1][0],data['a'][2][0]], 'size':[data['a'][0][1],data['a'][1][1],data['a'][2][1]]}],
                                    'timestamp': timestamp}
                
                self.spread_prices = self.predictor.get_target_spread_price(
                                        orderbook=self.orderbook,
                                        orderbook_5min= self.orderbook_5min,
                                        open_threshold=self.config.OPEN_THRESHOLD,
                                        stop_loss_threshold=self.config.STOP_LOSS_THRESHOLD,
                                    )
                
                if self.spread_prices \
                    and self.predictor.ref_spreads.is_warmed_up \
                    and self.predictor.target_spreads.is_warmed_up:
                    print("Time to create open orders")
                    #await self.pricer.create_open_orders(self.spread_prices)
                    await task_queue.put(
                            partial(self.pricer.create_open_orders,
                            self.predictor.spread_quotes)
                        )
                    self.remember_quotos = self.spread_prices
                

    async def Update_Trade(self,task_queue):
                ws = self.bm2.futures_socket()
                print("in update trade")
                async with ws as wscm:
                    while True:
                        resp = await wscm.recv()
                        print("resp :",resp)
                        if resp['e'] == "ORDER_TRADE_UPDATE":
                            if resp['o']['X'] == "FILLED" or resp['o']['X'] == "EXPIRED" or resp['o']['X'] == 'PARTIALLY_FILLED':
                                self.trades = resp['o']
                                #print(self.trades)
                                #print(resp['o'])
                                await task_queue.put(partial(self.pricer.manage_trade, self.trades, self.predictor.spread_quotes))
                        '''
                        if resp['e'] == "ACCOUNT_UPDATE" and resp['a']['P']:
                                for pos in resp['a']['P']:
                                    table= pretty_table(pos)
                                    #self.bot.send_message(chat_id=self.chat_id, text =f'<pre>{json.dumps(pos)}</pre>', parse_mode=telegram.ParseMode.HTML)
                                    self.bot.send_message(chat_id=self.chat_id, text = f'<pre>{table}</pre>', parse_mode=telegram.ParseMode.HTML)
                        '''
                            
                        
    async def execute_task(self, task_queue):
        while True:
            try:
                task = await task_queue.get()
                if asyncio.iscoroutinefunction(task.func):
                    await task()
                else:
                    task()
                task_queue.task_done()
            except Exception as e:
                print(traceback.format_exc())
    async def execute(self):
        while True:
            try:
                task_queue  = asyncio.Queue()
                trade_queue = asyncio.Queue()
                update_ob_ref = asyncio.create_task(self.Update_orderbook(task_queue,self.config.REFERENCE_SYMBOL))
                update_ob_target = asyncio.create_task(self.Update_orderbook(task_queue,self.config.TARGET_SYMBOL))
                update_trade = asyncio.create_task(self.Update_Trade(trade_queue))
                tasks = []
                tasks.append(asyncio.create_task(self.execute_task(trade_queue)))
                for i in range(2):
                    task = asyncio.create_task(self.execute_task(task_queue))
                    tasks.append(task)
                await asyncio.gather(
                    update_ob_ref, 
                    update_ob_target,
                    update_trade,
                    *tasks
                )
                # await task_queue.join()
                # await trade_queue.join()
            
            except Exception as e:
                #print(traceback.format_exc())
                self.log.info(
                    e)
                update_ob_ref.cancel()
                update_ob_target.cancel()
                update_trade.cancel()
                for t in tasks:
                    t.cancel()
                #continue
        


