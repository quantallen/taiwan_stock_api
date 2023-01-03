import sys
import asyncio
import os,sys
#設定當前工作目錄，放再import其他路徑模組之前
os.chdir(sys.path[0])
sys.path.append('./module')
from tw_pt import TF_pair_trading
import shioaji as sj
from config import Pair_Trading_Config
from credentials import person_id, account, password, c_path
async def main():
    api = sj.Shioaji()
    accounts = api.login(account, password)
    api.activate_ca(
        ca_path=c_path,
        ca_passwd= person_id,
        person_id= person_id,
    )
    configs = Pair_Trading_Config()
    tw_pair_trading = TF_pair_trading(api, api, configs)
    await tw_pair_trading.execute()
if __name__ == '__main__':
    asyncio.run(main())