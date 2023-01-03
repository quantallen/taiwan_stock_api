import sys

from pkg_resources import EntryPoint
sys.path.append('./simulator')


import string
from Trade import Trade

class Position:

    def __init__(self,
                 symbol : string,
                 side : string = None,
                 entry_price : float = 0.0,
                 size : float = 0.0,
                 value : float = 0.0) -> None:
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.size = size
        self.value = value

    def update(self, trade):
        if not self.side:
            self.side = trade.side
        
        size = trade.size if trade.side == "BUY" else -trade.size
        self.value += trade.price * size
        self.size += size
        self.entry_price = 0.0 if self.size == 0 else self.value / self.size

        if self.size > 0 :
            self.side = "BUY"
        elif self.size < 0:
            self.side = "SELL"
        else:
            self.side = None
            self.value = 0.0


class Positions:

    def __init__(self, ref_symbol, target_symbol):
        self.positions = {}
        self.positions[ref_symbol] = Position(ref_symbol)
        self.positions[target_symbol] = Position(target_symbol)
        

    def update(self, symbol, trade):
        self.positions[symbol].update(trade)

    def get_position(self, symbol):
        return self.positions[symbol]


def main():
    pos = Positions("BTC", "ETH")
    trade = Trade(id="", symbol="", side="BUY", base_currency=None, quote_currency=None, price=40000, size=1)
    pos.update("BTC", trade)


if __name__ == '__main__':
    main()


