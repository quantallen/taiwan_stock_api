from asyncio.log import logger
from pty import slave_open
from turtle import position
from attr import s
import numpy as np
import collections
import time
import PTwithTimeTrend_AllStock as ptm
import pandas as pd
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import os
from decimal import Decimal
import decimal


dtype = {
    'S1': str,
    'S2': str,
    'VECMQ': float,
    'mu': float,
    'Johansen_slope': float,
    'stdev': float,
    'model': int,
    'w1': float,
    'w2': float
}
CLOSE_POSITION = {
    "BUY": "SELL",
    "SELL": "BUY"
}
ADD_POS = 0.2

def makehash():
    return collections.defaultdict(makehash)


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
        return float(Decimal(x).quantize(PRECISION_AMOUNT))

class SpreadQuotes:
    spread_price = makehash()
    spread_size = makehash()
    spread_symbol = makehash()
    def __init__(self,ref_symbol,target_symbol):
        self.ref = ref_symbol
        self.target = target_symbol
        
    def set_size(self, symbol, size):
        #assert symbol in ["BTC-USD", "ETH-USD"]
        assert symbol in [self.ref, self.target]
        self.spread_size[symbol] = size

    def get_size(self, symbol):
        #assert symbol in ["BTC-USD", "ETH-USD"]
        assert symbol in [self.ref, self.target]
        return self.spread_size[symbol]

    def set_price(self, symbol, price):
        self.spread_price[symbol] = price

    def get_price(self, symbol):
        #assert symbol in ["BTC-USD", "ETH-USD"]
        assert symbol in [self.ref, self.target]
        return self.spread_price[symbol]

    def set_side(self, symbol, side):
        self.spread_symbol[symbol] = side

    def get_side(self, symbol):
        #assert symbol in ["BTC-USD", "ETH-USD"]
        assert symbol in [self.ref, self.target]
        return self.spread_symbol[symbol]


class Spreads:

    index = 0
    is_warmed_up = False

    def __init__(self, window_size):
        self.xs = np.zeros(window_size)
        self.window_size = window_size

    def update(self, x):

        if self.index == self.window_size:
            self.xs = shift(self.xs, -1, cval=0)
            self.index = 119
        self.xs[self.index % self.window_size] = x
        # print(self.xs)
        if self.index == self.window_size - 1:
            self.is_warmed_up = True
        self.index += 1


class Predictor:
    
    five_min_timestamp_1 = 0
    five_min_timestamp_2 = 0
    sec_timestamp_1 = 0
    sec_timestamp_2 = 0

    def __init__(self, window_size, ref_symbol, target_symbol, slippage,log, ref_trunc, target_trunc):
        self.window_size = window_size
        self.ref_symbol = ref_symbol
        self.target_symbol = target_symbol
        self.ref_spreads = Spreads(self.window_size)
        self.target_spreads = Spreads(self.window_size)
        self.ref_timestamp = 0
        self.target_timestamp = 0
        self.slippage = slippage
        self.spread_quotes = SpreadQuotes(self.ref_symbol,self.target_symbol)
        self.logger = log
        self.position = 0
        self.table = {
            "w1": 0,
            "w2": 0,
            "mu": 0,
            "stdev": 0,
            "model": 0,
            "capital": 2000
        }
        self.ref_size = 0
        self.target_size = 0
        self.close_ref_size = 0
        self.close_target_size = 0
        self.cointegration_check = False
        self.timestamp_check = False
        self.count = 0
        self.cointegration_upline = []
        self.cointegration_downline = []
        self.increase = ADD_POS
        self.ref_trunc = ref_trunc
        self.target_trunc = target_trunc
    def get_asks(self, orderbook):
        ref_ask = None
        target_ask = None
        if orderbook[self.ref_symbol]['sellQuote'] and orderbook[self.target_symbol]['sellQuote']:
            ref_ask = float(orderbook[self.ref_symbol]
                            ['sellQuote'][0]['price'])
            target_ask = float(
                orderbook[self.target_symbol]['sellQuote'][0]['price'])
        return ref_ask, target_ask

    def get_bids(self, orderbook):
        ref_bid = None
        target_bid = None
        if orderbook[self.ref_symbol]['buyQuote'] and orderbook[self.target_symbol]['buyQuote']:
            ref_bid = float(orderbook[self.ref_symbol]['buyQuote'][0]['price'])
            target_bid = float(
                orderbook[self.target_symbol]['buyQuote'][0]['price'])
        return ref_bid, target_bid
    def get_level_asks(self, orderbook):
        ref_ask = None
        target_ask = None
        if orderbook[self.ref_symbol]['sellQuote'] and orderbook[self.target_symbol]['sellQuote']:
            ref_ask = (float(orderbook[self.ref_symbol]['sellQuote'][0]['price'][0]) + float(orderbook[self.ref_symbol]['sellQuote'][0]['price'][1]) + float(orderbook[self.ref_symbol]['sellQuote'][0]['price'][2])) / 3
            target_ask = (float(orderbook[self.target_symbol]['sellQuote'][0]['price'][0]) + float(orderbook[self.target_symbol]['sellQuote'][0]['price'][1]) + float(orderbook[self.target_symbol]['sellQuote'][0]['price'][2])) / 3
        return ref_ask, target_ask

    def get_level_bids(self, orderbook):
        ref_bid = None
        target_bid = None
        if orderbook[self.ref_symbol]['buyQuote'] and orderbook[self.target_symbol]['buyQuote']:
            ref_bid = (float(orderbook[self.ref_symbol]['buyQuote'][0]['price'][0]) + float(orderbook[self.ref_symbol]['buyQuote'][0]['price'][1]) + float(orderbook[self.ref_symbol]['buyQuote'][0]['price'][2])) / 3

            target_bid = (float(orderbook[self.target_symbol]['buyQuote'][0]['price'][0]) + float(orderbook[self.target_symbol]['buyQuote'][0]['price'][1]) + float(orderbook[self.target_symbol]['buyQuote'][0]['price'][2])) / 3
        return ref_bid, target_bid

    def update_spreads(self, orderbook):
        #print(orderbook[self.ref_symbol]['timestamp'],self.ref_timestamp,orderbook[self.target_symbol]['timestamp'],self.target_timestamp)
        if self.ref_symbol in orderbook and self.target_symbol in orderbook and orderbook[self.ref_symbol]['timestamp'] != self.ref_timestamp and orderbook[self.target_symbol]['timestamp'] != self.target_timestamp:
            self.target_timestamp = orderbook[self.target_symbol]['timestamp']
            self.ref_timestamp = orderbook[self.ref_symbol]['timestamp']
            ref_ask, target_ask = self.get_asks(orderbook)
            ref_bid, target_bid = self.get_bids(orderbook)
            ref_mid_price = (ref_ask + ref_bid) / 2  # ref mid price
            target_mid_price = (target_ask + target_bid) / \
                2  # target mid price
            #print(datetime.now())
            print(f"ref :{ref_mid_price} , target : {target_mid_price}")
            if ref_ask and target_ask and ref_bid and target_bid:
                #ask_spread = target_ask - ref_ask
                #bid_spread = target_bid - ref_bid

                self.ref_spreads.update(ref_mid_price)
                self.target_spreads.update(target_mid_price)
                #print(f"ref_spread : {self.ref_spreads}")
                #print(f"target_spread : {self.target_spreads}")

    def cointegration_test(self,):
        #print("in cointegration")
        tmp = {self.ref_symbol: self.ref_spreads.xs,
               self.target_symbol: self.target_spreads.xs}
        price_series = [[r, t] for r, t in zip(
            self.ref_spreads.xs, self.target_spreads.xs)]
        price_series = np.array(price_series)
        #print("prices series",price_series)
        price_data = pd.DataFrame(tmp)
        #dailytable = ptm.formation_table(price_data,self.window_size)
        dailytable = ptm.refactor_formation_table(
            price_series, self.window_size)
        # btc_eth_table = pd.DataFrame(dailytable, columns=[
        #                             'S1', 'S2', 'VECM(q)', 'mu', 'Johansen_slope', 'stdev', 'model', 'w1', 'w2'],)
        # print(btc_eth_table)
        # if not btc_eth_table.empty:
        if len(dailytable) != 0:
            # print("yes")
            '''
            mean = pd.to_numeric(btc_eth_table["mu"], errors='coerce').astype(float)
            std = pd.to_numeric(btc_eth_table["stdev"], errors='coerce').astype(float)
            model = pd.to_numeric(btc_eth_table["model"], errors='coerce').astype('Int32')
            w1 = pd.to_numeric(btc_eth_table["w1"], errors='coerce').astype(float)
            w2 = pd.to_numeric(btc_eth_table["w2"], errors='coerce').astype(float)
            '''
            # return mean, std, model, w1, w2
            return dailytable[0], dailytable[1], [dailytable[2]], dailytable[3], dailytable[4]
        else:
            return [0], [0], [0], [0], [0]

    def slippage_number(self, x, size):
        neg = x * (-1)
        if self.position == -1:
            return neg if size > 0 else x
        elif self.position == 1:
            return neg if size < 0 else x

    def side_determination(self, size):
        if self.position == -1:
            return "SELL" if size > 0 else "BUY"
        elif self.position == 1:
            return "SELL" if size < 0 else "BUY"

    def open_Quotes_setting(self, ref_trade_price, target_trade_price):
        slippage = self.slippage
        
        self.ref_size, self.target_size = self.table["w1"] * self.table["capital"] / \
            ref_trade_price, self.table["w2"] * \
            self.table["capital"] / target_trade_price
        
        self.ref_size = trunc_amount_ref(self.ref_size,self.ref_trunc)
        self.target_size = trunc_amount_target(self.target_size,self.target_trunc)
        self.spread_quotes.set_price(
            self.ref_symbol, ref_trade_price * (1 + self.slippage_number(slippage, self.ref_size)))
        self.spread_quotes.set_price(
            self.target_symbol, target_trade_price * (1 + self.slippage_number(slippage, self.target_size)))
        self.spread_quotes.set_size(
            self.ref_symbol, abs(self.ref_size))
        self.spread_quotes.set_size(
            self.target_symbol, abs(self.target_size))
        self.spread_quotes.set_side(
            self.ref_symbol, self.side_determination(self.ref_size)
        )
        self.spread_quotes.set_side(
            self.target_symbol, self.side_determination(self.target_size)
        )
        self.close_ref_size += self.ref_size
        self.close_target_size += self.target_size
        print(f'reference_price = {ref_trade_price * (1 + self.slippage_number(slippage,self.ref_size))} . size = {abs(self.ref_size)} , side = {self.side_determination(self.ref_size)}')
        print(f'target_price = {target_trade_price *(1 + self.slippage_number(slippage,self.target_size))} . size = {abs(self.target_size)} , side = {self.side_determination(self.target_size)}')

    def close_Quotes_setting(self, ref_trade_price, target_trade_price):
        slippage = self.slippage

        # up -> size < 0 -> buy -> ask
        self.spread_quotes.set_price(
            self.ref_symbol, ref_trade_price * (1 - self.slippage_number(slippage, self.close_ref_size)))
        self.spread_quotes.set_price(
            self.target_symbol, target_trade_price * (1 - self.slippage_number(slippage, self.close_target_size)))
        self.spread_quotes.set_size(
            self.ref_symbol, abs(self.close_ref_size))
        self.spread_quotes.set_size(
            self.target_symbol, abs(self.close_target_size))
        self.spread_quotes.set_side(
            self.ref_symbol, CLOSE_POSITION[self.side_determination(
                self.close_ref_size)]
        )
        self.spread_quotes.set_side(
            self.target_symbol, CLOSE_POSITION[self.side_determination(
                self.close_target_size)]
        )
        print(f'reference_price = {ref_trade_price * (1 - self.slippage_number(slippage,self.close_ref_size))} . size = {abs(self.close_ref_size)} , side = {CLOSE_POSITION[self.side_determination(self.close_ref_size)]}')
        print(f'target_price = {target_trade_price *(1 - self.slippage_number(slippage,self.close_target_size))} . size = {abs(self.close_target_size)} , side = {CLOSE_POSITION[self.side_determination(self.close_target_size)]}')
        #self.position = 888
        self.position = 0
        self.close_ref_size = 0
        self.close_target_size = 0
    '''
    def draw_realtime_pic(self,open_threshold,stop_loss_threshold,stamp):
        
        path_to_image = "./real_time/"
        path = f'{path_to_image}{self.ref_symbol}_{self.target_symbol}_PIC/' 
        isExist = os.path.exists(path)
        if not isExist:    
            # Create a new directory because it does not exist 
            os.makedirs(path)
        print("The new directory is created!")
        curDT = datetime.now()
        time = curDT.strftime("%Y%m%d%H%M")
        sp =  self.table['w1'] * np.log(self.ref_spreads.xs) + self.table['w2'] * np.log(self.target_spreads.xs)
        fig, ax1 = plt.subplots(figsize=(20, 10))
        ax1.plot(sp, color='tab:blue', alpha=0.75)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.hlines(open_threshold * self.table['stdev'] + self.table['mu'], 0, len(sp) + 10,'b')
        ax1.hlines(stop_loss_threshold * self.table['stdev'] + self.table['mu'], 0, len(sp) + 10, 'b') 
        ax1.hlines(self.table['mu'] - open_threshold * self.table['stdev'], 0, len(sp) + 10,'b')
        ax1.hlines(self.table['mu'] - stop_loss_threshold * self.table['stdev'], 0, len(sp) + 10, 'b') 
        ax1.hlines(self.table['mu'], 0, len(sp) + 10, 'black') 
        ax1.scatter(len(sp)  ,stamp, color='g', edgecolors='r', marker='o')
        plt.savefig(path + str(self.ref_symbol) + '_' + str(self.target_symbol) + '_' +'realtime_spread.png')
    '''
    def draw_pictrue(self,open_threshold,stop_loss_threshold,stamp,POS):
        path_to_image = "./trading_position_pic/"
        path = f'{path_to_image}{self.ref_symbol}_{self.target_symbol}_PIC/' 
        isExist = os.path.exists(path)
        if not isExist:    
            # Create a new directory because it does not exist 
            os.makedirs(path)
        print("The new directory is created!")
        curDT = datetime.now()
        time = curDT.strftime("%Y%m%d%H%M")
        sp =  self.table['w1'] * np.log(self.ref_spreads.xs) + self.table['w2'] * np.log(self.target_spreads.xs)
        fig, ax1 = plt.subplots(figsize=(20, 10))
        ax1.plot(sp, color='tab:blue', alpha=0.75)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.hlines(open_threshold * self.table['stdev'] + self.table['mu'], 0, len(sp) + 10,'b')
        ax1.hlines(stop_loss_threshold * self.table['stdev'] + self.table['mu'], 0, len(sp) + 10, 'b') 
        ax1.hlines(self.table['mu'] - open_threshold * self.table['stdev'], 0, len(sp) + 10,'b')
        ax1.hlines(self.table['mu'] - stop_loss_threshold * self.table['stdev'], 0, len(sp) + 10, 'b') 
        ax1.hlines(self.table['mu'], 0, len(sp) + 10, 'black') 
        ax1.scatter(len(sp) + 1 ,stamp, color='g', edgecolors='r', marker='o')
        ax1.text(3,-3,f"w1 = {self.table['w1']}\nw2 = {self.table['w2']}\nstd = {self.table['stdev']}\nmu = {self.table['mu']}")
        if POS == 'open':
            plt.savefig(path + str(self.ref_symbol) + '_' + str(self.target_symbol) + '_' +POS+'spread_'+ time+'.png')
        elif POS == 'close':
            plt.savefig(path + str(self.ref_symbol) + '_' + str(self.target_symbol) + '_' +POS+'spread_'+ time+'.png')

    def get_target_spread_price(self, orderbook, orderbook_5min, open_threshold, stop_loss_threshold):
        if self.ref_spreads.is_warmed_up and self.target_spreads.is_warmed_up and orderbook[self.ref_symbol]['timestamp'] != self.sec_timestamp_1 and orderbook[self.target_symbol]['timestamp'] != self.sec_timestamp_2:
            ref_ask, target_ask = self.get_asks(orderbook_5min)
            ref_bid, target_bid = self.get_bids(orderbook_5min)
            ref_mid_price = (ref_ask + ref_bid) / 2  # ref mid price
            target_mid_price = (target_ask + target_bid) / 2
            self.sec_timestamp_1 = orderbook[self.ref_symbol]['timestamp']
            self.sec_timestamp_2 = orderbook[self.target_symbol]['timestamp']
            ref_ask, target_ask = self.get_level_asks(orderbook)
            ref_bid, target_bid = self.get_level_bids(orderbook)
            '''
            if self.position == 0 :
                print("FUUCK IN LA")
                self.position = -1
                self.ref_size = -0.123244 
                self.target_size = 0.0521321
                if self.ref_size < 0 and self.target_size > 0:
                            print("Allen is handsome")
                            self.open_Quotes_setting(ref_ask,target_bid)
                            print(ref_ask,target_bid)
                            return self.spread_quotes
            elif self.position == -1 and self.count == 10 :
                
                self.count = 0 
                if self.ref_size < 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid,target_ask)
                            print(ref_bid,target_ask)
                            return self.spread_quotes
            elif self.position == -1 :
                self.count += 1
            '''
        
            if self.five_min_timestamp_1 != orderbook_5min[self.ref_symbol]['timestamp'] and self.five_min_timestamp_2 != orderbook_5min[self.target_symbol]['timestamp']:
                self.five_min_timestamp_1 = orderbook_5min[self.ref_symbol]['timestamp']
                self.five_min_timestamp_2 = orderbook_5min[self.target_symbol]['timestamp']
                self.cointegration_check = False
                self.timestamp_check = True
            else:
                self.timestamp_check = False

            if self.position == 0 and self.cointegration_check is False and self.timestamp_check is True:
                print("in test cointegration")
                mu, stdev, model, w1, w2 = self.cointegration_test()
                if model[0] > 0 and model[0] < 4 and w1 * w2 < 0 :
                    self.cointegration_check = True
                    self.table = {
                        "w1": float(w1),
                        "w2": float(w2),
                        "mu": float(mu),
                        "stdev": float(stdev),
                        "model": model[0],
                        "capital": 2000
                    }
            if self.position == 0 and self.cointegration_check == True:
                self.increase = ADD_POS                
                if self.table["w1"] < 0 and self.table["w2"] > 0:
                    spread_stamp_up = self.table["w1"] * \
                        np.log(ref_ask) + \
                        self.table["w2"] * np.log(target_bid)
                    spread_stamp_down = self.table["w1"] * \
                        np.log(ref_bid) + \
                        self.table["w2"] * np.log(target_ask)

                elif self.table["w1"] > 0 and self.table["w2"] < 0:
                    spread_stamp_up = self.table["w1"] * \
                        np.log(ref_bid) + \
                        self.table["w2"] * np.log(target_ask)
                    spread_stamp_down = self.table["w1"] * \
                        np.log(ref_ask) + \
                        self.table["w2"] * np.log(target_bid)

                elif self.table["w1"] > 0 and self.table["w2"] > 0:
                    spread_stamp_up = self.table["w1"] * \
                        np.log(ref_bid) + \
                        self.table["w2"] * np.log(target_bid)
                    spread_stamp_down = self.table["w1"] * \
                        np.log(ref_ask) + \
                        self.table["w2"] * np.log(target_ask)

                elif self.table["w1"] < 0 and self.table["w2"] < 0:
                    spread_stamp_up = self.table["w1"] * \
                        np.log(ref_ask) + \
                        self.table["w2"] * np.log(target_ask)
                    spread_stamp_down = self.table["w1"] * \
                        np.log(ref_bid) + \
                        self.table["w2"] * np.log(target_bid)

                if spread_stamp_up > open_threshold * self.table['stdev'] + self.table['mu'] and spread_stamp_up < self.table["mu"] + self.table["stdev"] * stop_loss_threshold:

                    self.position = -1
                    self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_up,'open')
                    print(
                        f"上開倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                    if self.table['w1'] < 0 and self.table['w2'] > 0:
                        self.open_Quotes_setting(ref_ask, target_bid)
                        print(ref_ask, target_bid)
                        return self.spread_quotes
                    elif self.table['w1'] > 0 and self.table['w2'] < 0:
                        self.open_Quotes_setting(ref_bid, target_ask)

                        print(ref_bid, target_ask)
                        return self.spread_quotes
                    elif self.table['w1'] < 0 and self.table['w2'] < 0:
                        self.open_Quotes_setting(ref_ask, target_ask)

                        print(ref_ask, target_ask)
                        return self.spread_quotes
                    elif self.table['w1'] > 0 and self.table['w2'] > 0:
                        self.open_Quotes_setting(ref_bid, target_bid)

                        print(ref_bid, target_bid)
                        return self.spread_quotes

                elif spread_stamp_down < self.table['mu'] - open_threshold * self.table['stdev'] and spread_stamp_down > self.table["mu"] - self.table["stdev"] * stop_loss_threshold:
                    self.position = 1
                    self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_down,'open')
                    print(
                        f"下開倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                    print(f"Ref bid:{ref_bid} ; Target_ask : {target_ask}")
                    if self.table["w1"] < 0 and self.table['w2'] > 0:
                        self.open_Quotes_setting(ref_bid, target_ask)

                        print(ref_bid, target_ask)
                        return self.spread_quotes
                    elif self.table["w1"] > 0 and self.table['w2'] < 0:
                        self.open_Quotes_setting(ref_ask, target_bid)

                        print(ref_ask, target_bid)
                        return self.spread_quotes
                    elif self.table["w1"] < 0 and self.table['w2'] < 0:
                        self.open_Quotes_setting(ref_bid, target_bid)

                        print(ref_bid, target_bid)
                        return self.spread_quotes
                    elif self.table["w1"] > 0 and self.table['w2'] > 0:
                        self.open_Quotes_setting(ref_ask, target_ask)

                        print(ref_ask, target_ask)
                        return self.spread_quotes
            elif self.position != 0:

                if self.position == -1:
                    if self.ref_size < 0 and self.target_size > 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_bid) + \
                            self.table["w2"] * np.log(target_ask)
                        spread_stamp_up = self.table["w1"] * \
                        np.log(ref_ask) + \
                        self.table["w2"] * np.log(target_bid)

                    elif self.ref_size > 0 and self.target_size < 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_ask) + \
                            self.table["w2"] * np.log(target_bid)
                        spread_stamp_up = self.table["w1"] * \
                        np.log(ref_bid) + \
                        self.table["w2"] * np.log(target_ask)

                    elif self.ref_size > 0 and self.target_size > 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_ask) + \
                            self.table["w2"] * np.log(target_ask)
                        spread_stamp_up = self.table["w1"] * \
                        np.log(ref_bid) + \
                        self.table["w2"] * np.log(target_bid)
                    elif self.ref_size < 0 and self.target_size < 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_bid) + \
                            self.table["w2"] * np.log(target_bid)
                        spread_stamp_up = self.table["w1"] * \
                        np.log(ref_ask) + \
                        self.table["w2"] * np.log(target_ask)
                    if spread_stamp_up > (open_threshold + self.increase) * self.table['stdev'] + self.table['mu'] and spread_stamp_up < self.table["mu"] + self.table["stdev"] * stop_loss_threshold and self.increase < ADD_POS * 2:
                        self.increase += ADD_POS
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_up,'open')
                        if self.table['w1'] < 0 and self.table['w2'] > 0:
                            self.open_Quotes_setting(ref_ask, target_bid)
                            print(ref_ask, target_bid)
                            return self.spread_quotes
                        elif self.table['w1'] > 0 and self.table['w2'] < 0:
                            self.open_Quotes_setting(ref_bid, target_ask)

                            print(ref_bid, target_ask)
                            return self.spread_quotes
                        elif self.table['w1'] < 0 and self.table['w2'] < 0:
                            self.open_Quotes_setting(ref_ask, target_ask)

                            print(ref_ask, target_ask)
                            return self.spread_quotes
                        elif self.table['w1'] > 0 and self.table['w2'] > 0:
                            self.open_Quotes_setting(ref_bid, target_bid)
                            print(ref_ask, target_ask)
                            return self.spread_quotes    
                    elif spread_stamp < self.table['mu']:
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close')
                        print(
                            f"上開倉正常平倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        self.cointegration_check = False
                        if self.ref_size < 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid, target_ask)

                            print(ref_bid, target_ask)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_ask, target_bid)

                            print(ref_ask, target_bid)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_ask, target_ask)

                            print(ref_ask, target_ask)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_bid, target_bid)

                            print(ref_bid, target_bid)
                            return self.spread_quotes
                    elif spread_stamp > self.table["mu"] + self.table["stdev"] * stop_loss_threshold:
                        self.cointegration_check = False
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close')
                        print(
                            f"上開倉停損平倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        if self.ref_size < 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid, target_ask)

                            print(ref_bid, target_ask)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_ask, target_bid)

                            print(ref_ask, target_bid)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_ask, target_ask)

                            print(ref_ask, target_ask)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_bid, target_bid)

                            print(ref_bid, target_bid)
                            return self.spread_quotes
                elif self.position == 1:
                    if self.ref_size < 0 and self.target_size > 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_ask) + \
                            self.table["w2"] * np.log(target_bid)
                        spread_stamp_down = self.table["w1"] * \
                        np.log(ref_bid) + \
                        self.table["w2"] * np.log(target_ask)
                    elif self.ref_size > 0 and self.target_size < 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_bid) + \
                            self.table["w2"] * np.log(target_ask)
                        spread_stamp_down = self.table["w1"] * \
                        np.log(ref_ask) + \
                        self.table["w2"] * np.log(target_bid)

                    elif self.ref_size > 0 and self.target_size > 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_bid) + \
                            self.table["w2"] * np.log(target_bid)
                        spread_stamp_down = self.table["w1"] * \
                        np.log(ref_ask) + \
                        self.table["w2"] * np.log(target_ask)

                    elif self.ref_size < 0 and self.target_size < 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_ask) + \
                            self.table["w2"] * np.log(target_ask)
                        spread_stamp_down = self.table["w1"] * \
                        np.log(ref_bid) + \
                        self.table["w2"] * np.log(target_bid)
                    if spread_stamp_down < self.table['mu'] - (open_threshold + self.increase) * self.table['stdev'] and spread_stamp_down > self.table["mu"] - self.table["stdev"] * stop_loss_threshold and self.increase < ADD_POS * 2:
                        print(self.increase)
                        self.increase += ADD_POS
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_down,'open')
                        if self.table["w1"] < 0 and self.table['w2'] > 0:
                            self.open_Quotes_setting(ref_bid, target_ask)
                            print(ref_bid, target_ask)
                            return self.spread_quotes
                        elif self.table["w1"] > 0 and self.table['w2'] < 0:
                            self.open_Quotes_setting(ref_ask, target_bid)

                            print(ref_ask, target_bid)
                            return self.spread_quotes
                        elif self.table["w1"] < 0 and self.table['w2'] < 0:
                            self.open_Quotes_setting(ref_bid, target_bid)

                            print(ref_bid, target_bid)
                            return self.spread_quotes
                        elif self.table["w1"] > 0 and self.table['w2'] > 0:
                            self.open_Quotes_setting(ref_ask, target_ask)

                            print(ref_ask, target_ask)
                            return self.spread_quotes
                    elif spread_stamp > self.table['mu']:
                        self.cointegration_check = False
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close')
                        print(
                            f"下開倉正常平倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        if self.ref_size < 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_ask, target_bid)

                            print(ref_ask, target_bid)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_bid, target_ask)
                            print(ref_bid, target_ask)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid, target_bid)
                            print(ref_bid, target_bid)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_ask, target_ask)
                            print(ref_ask, target_ask)
                            return self.spread_quotes
                    elif spread_stamp < self.table["mu"] - self.table["stdev"] * stop_loss_threshold:
                        self.cointegration_check = False
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close')
                        print(
                            f"下開倉停損平倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        if self.ref_size < 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_ask, target_bid)
                            print(ref_ask, target_bid)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_bid, target_ask)
                            print(ref_bid, target_ask)
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid, target_bid)
                            print(ref_bid, target_bid)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_ask, target_ask)
                            print(ref_ask, target_ask)
                            return self.spread_quotes
        
            