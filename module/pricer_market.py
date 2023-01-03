import asyncio
import decimal
import re
from decimal import Decimal
import logging
import sys
import random
import string
from simulator.Order import Order
import telegram
sys.path.append('./trading-simulator/module')



# Price precision for submitting orders
chat_id = '-642791530'
bot = telegram.Bot(token=('5384131643:AAFd62LyZl5mfI-Tzd0c_xTUYRKcRWugWpc'))


def first_trade_alert(symbol, price, side, size):
    bot.send_message(
        chat_id=chat_id, text=f'Create First Order Transaction_Alert ! : Crypto : {symbol} , price : {price}, side : {side}, size :{size} ')


def reorder_trade_alert(symbol, price, side, size):
    bot.send_message(
        chat_id=chat_id, text=f'Reorder Transaction_Alert ! : Crypto : {symbol} , price : {price}, side : {side}, size :{size} ')


def manage_trade_alert(symbol, price, side, size):
    bot.send_message(
        chat_id=chat_id, text=f'Manage Trade Transaction_Alert ! : Crypto : {symbol} , price : {price}, side : {side}, size :{size} ')


def round_price_ref(x,PRECISION_PRICE):
    """
    There's probably a faster way to do this...
    """
    return float(Decimal(x).quantize(PRECISION_PRICE))


def trunc_amount_ref(x,PRECISION_AMOUNT):
    """
    There's probably a faster way to do this...
    """
    with decimal.localcontext() as c:
        #        c.rounding = 'ROUND_DOWN'
        return float(Decimal(x).quantize(PRECISION_AMOUNT))
def round_price_target(x,PRECISION_PRICE):
    """
    There's probably a faster way to do this...
    """
    return float(Decimal(x).quantize(PRECISION_PRICE))


def trunc_amount_target(x,PRECISION_AMOUNT):
    """
    There's probably a faster way to do this...
    """
    with decimal.localcontext() as c:
        #        c.rounding = 'ROUND_DOWN'
        return float(Decimal(x).quantize(PRECISION_AMOUNT))


def side_to_price(side, x):
    neg = x * (-1)
    if side == "BUY":
        return x
    elif side == "SELL":
        return neg


class TwoWayDict(dict):
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2


class Pricer:
    active_orders = {}
    open_close_mapping = TwoWayDict()

    def __init__(self, api, ref_symbol, target_symbol, logger,configs):
        self.api = api
        self.ref_symbol = ref_symbol
        self.target_symbol = target_symbol
        self.log = logger
        self.config = configs
        self.lock = asyncio.Lock()
        self.lock2 = asyncio.Lock()
    async def manage_trade(self, trade, spread_prices):
        async with self.lock2:
            order_tasks = []
            size = 0
            order_id = trade["i"]
            size = float(trade["l"])
            symbol = trade['s']
            
            self.log.fills(
                "BINANCE", trade["i"],
                trade['s'],
                trade['o'],
                trade['S'],
                trade['ap'],
                trade['l']
            )
            '''
            origin_size = spread_prices.get_size(symbol)
            if symbol == self.ref_symbol :
                origin_size = trunc_amount_ref(origin_size,self.config.PRECISION_AMOUNT_REF)
                price = spread_prices.get_price(symbol)
                side = spread_prices.get_side(symbol)
                price = round_price_ref(price * (1 + side_to_price(side, 0.005)),self.config.PRECISION_PRICE_REF)
                new_size = trunc_amount_ref(origin_size-size,self.config.PRECISION_AMOUNT_REF)
            if symbol == self.target_symbol :
                origin_size = trunc_amount_target(origin_size,self.config.PRECISION_AMOUNT_TARGET)
                price = spread_prices.get_price(symbol)
                side = spread_prices.get_side(symbol)
                price = round_price_target(price * (1 + side_to_price(side, 0.005)),self.config.PRECISION_PRICE_TARGET)
                new_size = trunc_amount_target(origin_size-size,self.config.PRECISION_AMOUNT_TARGET)
            print("交易 :",trade)
            if origin_size > size:
                    
                    print("======= cancel_origin order =========")
                    await self.api.futures_cancel_order(symbol=symbol,orderId = int(order_id))           
                    print("======= sufficient the size =========")
                    #manage_trade_alert(symbol,price,side, trunc_amount(origin_size - size))
                    
                    await self.api.futures_create_order(symbol=symbol, side=side , price=  price, quantity= new_size,  type='LIMIT', timeInForce="GTC", recvWindow=5000)
                     
                #order_tasks.append(self.api.futures_coin_modify_order(symbol=symbol, type="PRICE",
                ###                                                    value=round_price(price * (1 + side_to_price(side, 0.01))), origClientOrderId=order_id, quantity=size,
                  #                                                  timeInForce="GTC",
                  #                                                  recvWindow=5000))
                    
                #_result = await asyncio.gather(*order_tasks)
                
                #print('amend order result :\n', _result)
            '''
    async def create_open_orders(self, spread_prices):
        print("===== create open orders =====")
        async with self.lock:
            order_tasks = []

            order_key_sell = "open_SELL_{}".format(0)
            order_key_buy = "open_BUY_{}".format(0)
            price = spread_prices.get_price(self.ref_symbol)
            price = round_price_ref(price,self.config.PRECISION_PRICE_REF)
            size = spread_prices.get_size(self.ref_symbol)
            size = trunc_amount_ref(size,self.config.PRECISION_AMOUNT_REF)
            side = spread_prices.get_side(self.ref_symbol)
            #first_trade_alert(self.ref_symbol, price, side, size)
            print("price and size :", price, size)
            if side == 'BUY':
                #order_tasks.append(self.api.futures_create_order(
                #    symbol=self.ref_symbol, side="BUY", price=price, quantity=size, newClientOrderId=order_key_buy, type='LIMIT', timeInForce="GTC", recvWindow=5000))
                order_tasks.append(self.api.futures_create_order(
                    symbol=self.ref_symbol, side="BUY", quantity=size, newClientOrderId=order_key_buy, type= "MARKET"))
            elif side == 'SELL':
                #order_tasks.append(self.api.futures_create_order(
                #    symbol=self.ref_symbol, side="SELL", price=price, quantity=size, newClientOrderId=order_key_buy, type='LIMIT', timeInForce="GTC", recvWindow=5000))
                order_tasks.append(self.api.futures_create_order(
                    symbol=self.ref_symbol, side="SELL", quantity=size, newClientOrderId=order_key_buy, type='MARKET'))
            price = spread_prices.get_price(self.target_symbol)
            price = round_price_target(price,self.config.PRECISION_PRICE_TARGET)
            size = spread_prices.get_size(self.target_symbol)
            size = trunc_amount_target(size,self.config.PRECISION_AMOUNT_TARGET)
            side = spread_prices.get_side(self.target_symbol)
            print("price and size :", price, size)
            if side == 'BUY':
                #order_tasks.append(self.api.futures_create_order(
                   # symbol=self.target_symbol, side="BUY", price=price, quantity=size, newClientOrderId=order_key_buy, type='LIMIT', timeInForce="GTC", recvWindow=5000))
                order_tasks.append(self.api.futures_create_order(
                    symbol=self.target_symbol, side="BUY",  quantity=size, newClientOrderId=order_key_buy, type='MARKET'))
            elif side == 'SELL':
                #order_tasks.append(self.api.futures_create_order(
                 #   symbol=self.target_symbol, side="SELL", price=price, quantity=size, newClientOrderId=order_key_buy, type='LIMIT', timeInForce="GTC", recvWindow=5000))
                order_tasks.append(self.api.futures_create_order(
                    symbol=self.target_symbol, side="SELL", quantity=size, newClientOrderId=order_key_buy, type='MARKET'))

            result = await asyncio.gather(*order_tasks)
            #print('order result :\n', result)

