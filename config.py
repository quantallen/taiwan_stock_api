from decimal import Decimal

class Pair_Trading_Config:
    # Target symbol is where the LIMIT orders are placed based on calculated spreads.
    REFERENCE_SYMBOL = "2330"
    FUTURE_REF_SYMBOL = "QXF202301"    
    TARGET_SYMBOL = "2303"
    FUTURE_TARGET_SYMBOL = "DAF202301"

    # Reference symbol is where the MARKET orders are intiated AFTER target symbol's limit orders are filled.
    
   #TARGET_SYMBOL = "ETH_USDT"

    # Reference symbol is where the MARKET orders are intiated AFTER target symbol's limit orders are filled.
    #REFERENCE_SYMBOL = "BTC_USDT"

    OPEN_THRESHOLD = 1.5

    STOP_LOSS_THRESHOLD = 5
    # Window size for calculating spread mean.
    MA_WINDOW_SIZE = 80
    
    RETRY_TIME = 1
    
    PRECISION_AMOUNT_REF = Decimal('0')
    
    PRECISION_PRICE_REF = Decimal('0.00')
    
    
    PRECISION_AMOUNT_TARGET = Decimal('0')
    
    PRECISION_PRICE_TARGET = Decimal('0.00')
    
    SLIPPAGE = 0.001
    TEST_SECOND = 60
    