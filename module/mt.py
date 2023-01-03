# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 23:57:28 2020

@author: John Hao Han
"""

import pandas as pd
import numpy as np
from scipy.linalg import eigh
import math
from integer import num_weight

def simp_frac(x,y,Range):
    #輸入兩個實數，可以找到最接近的整數比例
    #Range是限制，回傳的值再不考慮正負號的情況下只會在[1,Range]到[Range,1]之間
    
    #紀錄正負號
    PNlog1 , PNlog2 = (x>0)*2-1 , (y>0)*2-1
    #將兩個數字都變為正數
    intx , inty = x * PNlog1 , y * PNlog2
    
    #建構xy數列
    intrange = [i for i in range(1,Range+1)]  
    #依照xy數列建構空的atan夾角空間
    anglespace = np.zeros([ len(intrange) , len(intrange) ])
    
    for inti in range(len(intrange)):
        for intj in range(len(intrange)):
            #填充atan夾角空間，單位為弧度
            anglespace[inti,intj] = math.atan( intrange[intj] / intrange[inti] ) 
    
    # 兩個度數相減得到夾角弧度，degspace為夾角弧度的絕對值
    degspace = abs(anglespace - math.atan(inty/intx))
    # 找出最小夾角並回傳空間內座標
    optij = np.where(degspace == np.min(degspace) )
    
    # 依據空間內座標找出最適的xy，如果有重複答案選取最小，最後把前面的正負號乘回去
    SFx1 = intrange[optij[0][0]] * PNlog1
    SFy2 = intrange[optij[1][0]] * PNlog2
    
    return [SFx1,SFy2]



def Binal_comb(pool):
    # Binal_comb(range(3)) = [[0,1],[0,2],[1,2]] dytp=narray
    # 排列組合不重複的數列，專門處理Cn取2
    n = len(pool)
    stemp = []
    if 2 > n:
        return 'Error:Beasue input less 2 iter'
    else:
        stepi = 0
        stepj = 0
        for stepi in range(n):
            for stepj in range(stepi):
                stemp.append([pool[stepj],pool[stepi]])
        stemp.sort()
        result = np.array(stemp)
        return result

def VAR_model( raw_y , p ):
    
    #k = len(y.T)     # 幾檔股票
    n = len(raw_y)       # 資料長度
    # 以下，把資料疊好，準備做OLS估計
    xt = raw_y[:-p,:]
    for j in range(1,p):
        xt_1 = raw_y[j:-p+j,:]
        xt = np.hstack((xt_1,xt))
    
    int_one = np.ones((n-p, 1))
    int_trd = np.arange(1,n-p+1,1).reshape(n-p,1)
    insept_x = np.hstack( (  int_one ,  int_trd) )
    xt = np.hstack( (insept_x, xt) )
    
    yt = np.delete(raw_y,np.s_[0:p],axis=0)
    #資料疊好了，下面一行是轉成matrix，計算比較不會錯
    xt ,yt = np.mat(xt) , np.mat(yt)

    beta = ( xt.T * xt ).I * xt.T * yt                      # 計算VAR的參數
    E = yt - xt * beta                                      # 計算殘差
    Res_sigma = ( (E.T) * E ) / (n-p)                           # 計算殘差的共變異數矩陣
        
    return [  beta ,Res_sigma ]

# 配適 VAR(P) 模型 ，並利用BIC選擇落後期數，max_p意味著會檢查2~max_p--------------------
def order_select( raw_y , max_p ):
    k = len(raw_y.T)     # 幾檔股票
    n = len(raw_y)       # 資料長度
    lags = [i+1 for i in range(1, max_p)]  #產生一個2~max_p的list
    bic = np.zeros((len(lags),1))
    
    for p in range(len(lags)):
        sigma = VAR_model( raw_y , lags[p] )[1] 
        bic[p] = np.log( np.linalg.det(sigma) ) + np.log(n) * p * (k*k) / n
        
    bic_order = lags[np.argmin(bic)]
    
    return bic_order


# Johanse test與VECM模型參數-------------
def JCItestModel5( Xt , opt_q , alpha ):
    # 防呆檢查
    if alpha == 0.05:
        testCi = 0
    elif alpha == 0.01:
        testCi = 1
    else:
        return 'alpha should be 0.05 or 0.01.'
    
    [NumObs,NumDim] = Xt.shape
    
    if NumObs < NumDim:
        return 'Xt must be a T*NumDim matrix and T>NumDim. '
    elif NumDim > 2:
        return 'Only 2 NumDim can work.'
    elif opt_q < 1:
        return 'Set x higher than 1 '
    
    #最適落後期數VAR(p)應該要先用order_select 的BIC找最適p，且VEC(q)，q=p-1
    #Xt=NumObs*NumDim的矩陣，NumObs為觀察值（t=1,2,...,NumObs），NumDim為股票個數

    #設定計算用參數
    p = opt_q+1
    T = NumObs-p
    dY_ALL = Xt[1:,:]-Xt[0:-1,:]
    dY = dY_ALL[p-1:,:]
    Ys = Xt[p-1:-1,:]
    dX = np.zeros([T,NumDim*(p-1)])
    
    for xi in range(p-1):
        dX[:,xi*NumDim:(xi+1)*NumDim] = dY_ALL[p-xi-2:NumObs-xi-2,:]

    #H5增加截距與時間趨勢
    dX = np.hstack( ( dX, np.ones((T,1)) , np.arange(1,T+1,1).reshape(T,1) ) )
    '''
    #H4增加截距與時間趨勢
    dX = np.hstack( ( dX, np.ones((T,1)) ) )
    Ys = np.hstack( ( Ys, np.arange(1,T+1,1).reshape(T,1)))
    '''
    #轉成matrix，計算比較直觀
    dX , dY , Ys = np.mat(dX) , np.mat(dY), np.mat(Ys)
    
    #先求dX'*dX 方便下面做inverse
    DX = dX.T * dX
    #I-dX * (dX'*dX)^-1 * dX'
    M = np.identity(T)-dX*DX.I*dX.T
    
    #matrix 跟numpy 都可以用下面這行，跟上面是同樣的計算
    #M = np.identity(T) - np.dot(np.dot(dX , np.linalg.inv(np.dot(dX.T , dX))) , dX.T )
    
    R0 , R1= dY.T * M , Ys.T * M
    S00 = R0 * R0.T / T
    S01 = R0 * R1.T / T
    S10 = R1 * R0.T / T
    S11 = R1 * R1.T / T
    
    eigVals, eigvecs = eigh(S10 * S00.I * S01 , S11 , eigvals_only=False)
    '''
    #如果要驗算特徵值與特徵矩陣，MustBeZero越接近0越好
    eig_d = np.zeros((NumDim,NumDim))
    for e in range(NumDim):
        eig_d[e,e] = eig_D[e]
    
    MustBeZero = np.dot(Sxx , eig_V)-np.dot(np.dot(S11 , eig_V), eig_d)
    '''
    #排序特徵向量與特徵值
    sort_ind = np.argsort(-eigVals)
    eigVals = eigVals[sort_ind] 
    eigVecs = eigvecs[:,sort_ind]
    eigVals = eigVals.reshape(NumDim,1)
    
    #VECM各項參數
    JCIstat	= [[0]*11 for i in range(NumDim)]
    JCIstat[0][:]= ['','A','B','c0','d0','c1','d1','Bq','eigValue','eigVector','testStat']
    
    #是否通過檢定，True=通過，False=沒通過
    H = [[0]*2 for i in range(NumDim)]
    
    #CVTableRow 跟CVTable 都是算pValue的表格，但是因為CVTable表不精確，Pvalue等之後比較精確再來做
    #CVTableRow = [95, 99]  95=95% , 99=99%
    #CVTable = [ (k-r) , criterion percentage]
    CVTable = [[3.8415, 6.6349],
               [18.3969, 23.1574],
               [35.0131, 41.0722]]
    
    for rn in range(1,NumDim):
        B = np.mat(eigVecs[:,0:rn])
        A = S01*B
        W = dY-Ys*B*A.T
        P = dX.I * W # [B1,...,Bq]
        P = P.T
        c = P[:,-2]
        c0 = A.I * c
        c1 = c - A * c0
        d = P[:,-1]
        d0 = A.I * d
        d1 = d - A * d0
        
        Bq = [[0]*2 for i in range(opt_q)]
        for bi in range(1,opt_q+1):
            Bq[bi-1][0] = 'B'+str(bi)
            Bq[bi-1][1] = P[:,((bi-1)*NumDim):bi*NumDim]
            
        JCIstat[rn][0] = ['r'+str(rn)]
        JCIstat[rn][1] = A[:,0]
        JCIstat[rn][2] = B[:,0]
        JCIstat[rn][3] = c0
        JCIstat[rn][4] = d0
        JCIstat[rn][5] = c1
        JCIstat[rn][6] = d1
        JCIstat[rn][7] = Bq
        JCIstat[rn][8] = eigVals[rn,:]
        JCIstat[rn][9] = eigVecs[:,rn]
        #如果需要Residual = W - dX * P.T
        eig_lambda = np.cumprod(1-eigVals[rn:NumDim,:])
        JCIstat[rn][10] = -2 * np.log(eig_lambda[-1] ** (T/2))
        
        H[rn][0] = ['h'+str(rn)]
        H[rn][1] = CVTable[NumDim-rn-1][testCi] < JCIstat[rn][10]
        '''
        #因為統計檢定量的分配數字還不夠精確，Pvalue的計算暫緩
        CVraw = CVTable[NumDim-rn-1] 
        pvalue = np.interp(JCIstat[rn][11], CVraw , CVTableRow)
        pvalue = 1-(pvalue/100)
        '''
        return [H, JCIstat]


# Johanse test與VECM模型參數-------------
def JCItestModel1(Xt, opt_q, alpha):
    # 防呆檢查
    if alpha == 0.05:
        testCi = 0
    elif alpha == 0.01:
        testCi = 1
    else:
        return 'alpha should be 0.05 or 0.01.'

    [NumObs, NumDim] = Xt.shape

    if NumObs < NumDim:
        return 'Xt must be a T*NumDim matrix and T>NumDim. '
    elif NumDim > 2:
        return 'Only 2 NumDim can work.'
    elif opt_q < 1:
        return 'Set x higher than 1 '

    # 最適落後期數VAR(p)應該要先用order_select 的BIC找最適p，且VEC(q)，q=p-1
    # Xt=NumObs*NumDim的矩陣，NumObs為觀察值（t=1,2,...,NumObs），NumDim為股票個數

    # 設定計算用參數
    p = opt_q + 1
    T = NumObs - p
    dY_ALL = Xt[1:, :] - Xt[0:-1, :]
    dY = dY_ALL[p - 1:, :]
    Ys = Xt[p - 1:-1, :]
    dX = np.zeros([T, NumDim * (p - 1)])

    for xi in range(p - 1):
        dX[:, xi * NumDim:(xi + 1) * NumDim] = dY_ALL[p - xi - 2:NumObs - xi - 2, :]

    # 轉成matrix，計算比較直觀
    dX, dY, Ys = np.mat(dX), np.mat(dY), np.mat(Ys)

    # 先求dX'*dX 方便下面做inverse
    DX = dX.T * dX
    # I-dX * (dX'*dX)^-1 * dX'
    M = np.identity(T) - dX * DX.I * dX.T

    # matrix 跟numpy 都可以用下面這行，跟上面是同樣的計算
    # M = np.identity(T) - np.dot(np.dot(dX , np.linalg.inv(np.dot(dX.T , dX))) , dX.T )

    R0, R1 = dY.T * M, Ys.T * M
    S00 = R0 * R0.T / T
    S01 = R0 * R1.T / T
    S10 = R1 * R0.T / T
    S11 = R1 * R1.T / T

    eigVals, eigvecs = eigh(S10 * S00.I * S01, S11, eigvals_only=False)
    '''
    #如果要驗算特徵值與特徵矩陣，MustBeZero越接近0越好
    eig_d = np.zeros((NumDim,NumDim))
    for e in range(NumDim):
        eig_d[e,e] = eig_D[e]

    MustBeZero = np.dot(Sxx , eig_V)-np.dot(np.dot(S11 , eig_V), eig_d)
    '''
    # 排序特徵向量與特徵值
    sort_ind = np.argsort(-eigVals)
    eigVals = eigVals[sort_ind]
    eigVecs = eigvecs[:, sort_ind]
    eigVals = eigVals.reshape(NumDim, 1)

    # VECM各項參數
    JCIstat = [[0] * 7 for i in range(NumDim)]
    JCIstat[0][:] = ['', 'A', 'B', 'Bq', 'eigValue', 'eigVector', 'testStat']

    # 是否通過檢定，True=通過，False=沒通過
    H = [[0] * 2 for i in range(NumDim)]

    # CVTableRow 跟CVTable 都是算pValue的表格，但是因為CVTable表不精確，Pvalue等之後比較精確再來做
    # CVTableRow = [95, 99]  95=95% , 99=99%
    # CVTable = [ (k-r) , criterion percentage]
    CVTable = [[4.1475, 6.9701],
               [12.3329, 16.2917],
               [24.3168, 29.6712]]

    for rn in range(1, NumDim):
        B = np.mat(eigVecs[:, 0:rn])
        A = S01 * B
        W = dY - Ys * B * A.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T

        Bq = [[0] * 2 for i in range(opt_q)]
        for bi in range(1, opt_q + 1):
            Bq[bi - 1][0] = 'B' + str(bi)
            Bq[bi - 1][1] = P[:, ((bi - 1) * NumDim):bi * NumDim]

        JCIstat[rn][0] = ['r' + str(rn)]
        JCIstat[rn][1] = A[:, 0]
        JCIstat[rn][2] = B[:, 0]
        JCIstat[rn][3] = Bq
        JCIstat[rn][4] = eigVals[rn, :]
        JCIstat[rn][5] = eigVecs[:, rn]
        # 如果需要Residual = W - dX * P.T
        eig_lambda = np.cumprod(1 - eigVals[rn:NumDim, :])
        JCIstat[rn][6] = -2 * np.log(eig_lambda[-1] ** (T / 2))

        H[rn][0] = ['h' + str(rn)]
        H[rn][1] = CVTable[NumDim - rn - 1][testCi] < JCIstat[rn][6]
        '''
        #因為統計檢定量的分配數字還不夠精確，Pvalue的計算暫緩
        CVraw = CVTable[NumDim-rn-1] 
        pvalue = np.interp(JCIstat[rn][11], CVraw , CVTableRow)
        pvalue = 1-(pvalue/100)
        '''
        return [H, JCIstat]


def TimeSpreadMeanStd(St, JCIpara, cw, opt_q , OpenDel):
    '''
    #Debug用的參數
    St, JCIpara, cw= rowLS  , JCIstat , np.mat(CapitW[:,mi])
    opt_q,   OpenDel = opt_p-1 , OpenDrop
    '''
    cw = np.mat(cw)
    
    A = JCIpara[1][1]
    Beta = JCIpara[1][2]
    C0 = JCIpara[1][3]
    D0 = JCIpara[1][4]
    C1 = JCIpara[1][5]
    D1 = JCIpara[1][6]
    Bq = JCIpara[1][7]
    
    S_hat = np.zeros([St.shape[0],2])
    S_hat[0:opt_q+1,:] = St[0:opt_q+1,:]
    S_hat = np.mat(S_hat)
    
    # tf = 下面計算用的t
    tf = np.arange(-OpenDel+1,St.shape[0]-OpenDel+1)
    # 轉成matrix 方便計算
    tf = np.mat(tf)
    
    for si in range(opt_q+1,St.shape[0]):
        '''
        yt = yt-1 + dy
        dy = A(Beta'*yt-1 +  c0  + d0t ) + c1 + d1t + DX 
        dy = A*Beta'*yt-1 + A*c0 + A*d0t + c1 + d1t + DX 
        '''
        # m1 = yt-1 * Beta * A'
        m1 = S_hat[si-1,:] * Beta * A.T
        # m2 = A*c0 + A*d0t
        m2 = C0.T * A.T + tf[:,si] * (D0 * A.T)
        # m3 = c1 + d1t
        m3 = C1.T + tf[:,si] * D1.T
        
        #這個迴圈是在算DX
        dXt = np.matrix([0,0])
        for bi in range(opt_q):
            dXt = dXt + (S_hat[si-bi-1,:] - S_hat[si-bi-2,:] ) * Bq[bi][1]
        
        # yt = yt-1 + dy
        S_hat[si,:] = S_hat[si-1,:] + m1 + m2 + m3 + dXt
    
    # 迭代後的均值  
    Mean_Trend = S_hat * cw.T
    # 共整合序列
    Cointer_Spread = St * cw.T
    
    '''
    #畫個圖檢查一下
    firstP = 0
    plotx = np.arange(firstP,indataNum)
    plt.plot(plotx,Cointer_Spread,Mean_Trend)
    
    '''
    
    # 將均值兩次差分
    Mean_Trend_d2 = np.zeros([St.shape[0],1])
    Mean_Trend_d2[2:] = np.diff(Mean_Trend,n=2,axis=0)
    Mean_Trend_d2[0:2] = Mean_Trend_d2[2]
    smsp = np.full([1,1], np.nan)
    # 兩次差分後夠接近0的就視為收斂
    for si in range(St.shape[0]):
        if abs(Mean_Trend_d2[si]) < 0.000001:
            smsp = si
            break
    
    if np.isnan(smsp):
        StMean = np.zeros([St.shape[0],1])
        St_Std = 0
        smsp = St.shape[0]
    else:
        StMean = Mean_Trend
        ErrorT = Mean_Trend[smsp: ,]-Cointer_Spread[smsp: ,]
        St_Std = np.sqrt(ErrorT.T * ErrorT / len(Mean_Trend[smsp: ,]))

    return [StMean, St_Std, smsp]

def ADFtest_TR(Y, lagP , alpha):
    # 防呆檢查
    if not(alpha == 0.05 or alpha == 0.01):
        return 'alpha should be 0.05 or 0.01.'

    
    Y = Y.T
    N= Y.shape[0]
    T = N-(lagP+1)
    
    # CVTable [ effective sample sizes , Alpha ]
    # CVTable 為ADF-t分配的表格，底下檢定是否過門檻要利用這個表做內插，求得門檻值
    CVTable = pd.read_csv('D:\HSINHUA\PTSTrend\PythonCode\AdfTable.csv',header=None)
    alphaLevel = CVTable.iloc[0,:]
    CVTable =  CVTable.drop([0], axis=0)
    CVTable = CVTable.reset_index(drop=True)
    CVTable = CVTable.to_numpy()
    sampSizes = np.asarray([10,15,20,25,30,40,50,75,100,150,200,300,500,1000,10000])
    sampMaxMin = []
    sampMaxMininx = []
    
    # 找出T在sampSizes中對應的位置
    for si in range(sampSizes.shape[0]):
        if T < sampSizes[si] :
            sampMaxMininx.append(si-1)
            sampMaxMininx.append(si)
            sampMaxMin.append(sampSizes[si-1])
            sampMaxMin.append(sampSizes[si])
            break
    #這邊開始計算ADF，接著開始疊資料，方便後面的估計
    Y_lags = np.full([N,lagP+2], np.nan) 
    for yi in range (lagP+2):
        Y_lags[yi: , yi] = Y[0:N-yi]
    
    testY = Y_lags[(lagP+1):,0]
    dY_lags = -np.diff(Y_lags,n=1,axis=1)
    
    adf_X = np.hstack( (np.ones((N-(lagP+1),1)) , np.arange(1,T+1,1).reshape(T,1) ) )
    adf_X = np.hstack( (adf_X , Y_lags[(lagP+1):,1].reshape(T,1) , dY_lags[ (lagP+1): ,1:]) )
    
    #資料疊完，開始算，不用OLS估計，這邊用QR分解
    #定義 ADF 模型為：yt = c + δt + βyt−1 + ϕp dyt-p + et
    Q, R = np.linalg.qr(adf_X)
    #轉成Matrix型態方便計算
    Q, R, testY = np.mat(Q) , np.mat(R) , np.mat(testY)
    Beta = R.I * (Q.T * testY.T)
    Y_hat = np.dot(adf_X,Beta)
    adf_Res = testY.transpose() - Y_hat
    adf_SSE = adf_Res.T * adf_Res
    degreFedom = T-len(Beta)
    adf_MSE = adf_SSE/degreFedom
    S = R.I * np.mat(np.eye(len(Beta)))
    adf_Cov = float(adf_MSE) * S * S.T
    #se[0]=截距項c的se，se[1]=趨勢項δt的se，se[2]=Beta項的se，se[3]=ϕ的se
    se = np.sqrt(np.diag(adf_Cov))
    #獲得t值，等等做ADF-t檢定
    tValue = (Beta[2]-1) / se[2]
    
    #依據給定alpha抽取門檻值
    #線性內插求門檻值 tCvalue
    nc = (T-sampMaxMin[0]) / (sampMaxMin[1] -sampMaxMin[0])
    fv = CVTable[sampMaxMininx, np.where(alphaLevel == alpha)].T
    tCvalue = nc*(fv[1,0]-fv[0,0])+fv[0,0]
    
    if tValue < tCvalue:
        H=1
    else:
        H=0
    
    return H #[H, tValue, tCvalue]



def TradeCost(SSt, InitB, c ,LS):
    TC = 0
    if LS == 'S':
        TC = max(SSt[0]*InitB[0]*c,0)
        TC = TC + max(SSt[1]*InitB[1]*c,0)
        return TC
    elif LS == 'L':
        TC = abs(min(SSt[0]*InitB[0]*c,0))
        TC = TC + abs(min(SSt[1]*InitB[1]*c,0))
        return TC
def tax(payoff,rate):
    tax_price = payoff * (1 - rate * (payoff > 0))
    return tax_price    

def trade_up(spread , trend , Row_ST ,Ibeta , trStd , trCost , tros, trcs, trMtp):
    #spread:共整合序列 , trend:時間趨勢均值 , RowST:原始股價
    #Ibeta:整數化股票張數 , trStd:建模期間標準差 , trCost:交易成本
    #tros:開倉倍數, trcs:停損倍數
    '''    
    #Debug用
    spread, trend = cy[inNum:DailyNum+1,pi] ,cy_mean[inNum:DailyNum+1,pi]
    Row_ST= Smin[inNum:DailyNum+1,[int(OMinx[pi,0]), int(OMinx[pi,1])]] 
    Ibeta ,trStd, trCost ,tros , trcs= IntegerB[:,pi], SStd ,Cost ,Os, Fs
    trMtp = Max_tp
    '''
    #[總獲利,平倉獲利,停損獲利,換日強停獲利,換日強停虧損]
    Profit = np.zeros((1,5))
    #[開倉次數,平倉次數,停損次數,換日強停獲利次數,換日強停虧損次數]
    Count = np.zeros((1,5))
    OpenTrend = trend - trStd*tros #下開倉
    StopTrend = trend - trStd*trcs
    Position = 0 # 部位控制
    LogTradeTime = np.zeros((1,spread.shape[0])) # 時間紀錄
    openP = 0
    ForceP = 0
    opencount = 0
    opentime = 0
    closetime = 0
    LongOrShort = 1
    opens1payoff = 0
    opens2payoff = 0
    closes1payoff = 0
    closes2payoff = 0
    capital = 50000000
    for ti in range(spread.shape[0]):
        #尾盤的強制平倉處理
        if ti == spread.shape[0]-1:
            #若有倉則強制平倉
            if Position == 1: 
                #ForceP = np.dot(Row_ST[ti,:],Ibeta) - TradeCost(Row_ST[ti,:],Ibeta,trCost,'S') - openP
                closes1payoff = LongOrShort * Row_ST[97,0] * Ibeta[0]
                closes2payoff = LongOrShort * Row_ST[97,1] * Ibeta[1]
                ForceP = tax(closes1payoff,trCost)+tax(closes2payoff,trCost)+openP
                if ForceP > 0:
                    Profit[0,3] = ForceP
                    Count[0,3] = Count[0,3] + 1
                    Position = 0
                    LogTradeTime[0,ti] = 3
                    closetime = 97
                elif ForceP <= 0:
                    Profit[0,4] = ForceP
                    Count[0,4] = Count[0,4] + 1
                    Position = 0
                    LogTradeTime[0,ti] = 3
                    closetime = 97
        #尾盤前的交易
        else:
            if opencount <= 1 :
            #到期前若碰到平倉門檻且有倉，平倉
                if Position == 1 and spread[ti]>=trend[ti] and ti < 97:
                    #Profit[0,1] = np.dot(Row_ST[ti+1,:],Ibeta) - TradeCost(Row_ST[ti+1,:],Ibeta,trCost,'S') - openP
                    closes1payoff = LongOrShort * Row_ST[ti,0] * Ibeta[0]
                    closes2payoff = LongOrShort * Row_ST[ti,1] * Ibeta[1]
                    ForceP = tax(closes1payoff,trCost)+tax(closes2payoff,trCost)+openP
                    #Profit[0,1] = np.dot(Row_ST[ti,:],Ibeta) - TradeCost(Row_ST[ti,:],Ibeta,trCost,'S') - openP
                    Profit[0,1] = ForceP
                    Count[0,1] = Count[0,1] + 1
                    Position = 0
                    LogTradeTime[0,ti] = -1
                    closetime = ti
                #到期前若碰到停損門檻且有倉，停損    
                elif Position == 1 and spread[ti]<=StopTrend[ti]:
                    #Profit[0,2] = np.dot(Row_ST[ti+1,:],Ibeta) - TradeCost(Row_ST[ti+1,:],Ibeta,trCost,'S') - openP
                    Profit[0,2] = np.dot(Row_ST[ti,:],Ibeta) - TradeCost(Row_ST[ti,:],Ibeta,trCost,'S') - openP
                    Count[0,2] = Count[0,2] + 1
                    Position = -1
                    LogTradeTime[0,ti] = -2
                    closetime = ti
                #強制每配對至多開倉一次
                
                #到期前，若碰到開倉門檻且無倉，開倉    
                elif Position == 0 and spread[ti]<=OpenTrend[ti] and ti<(trMtp-150) and opencount != 1:
                    #openP = np.dot(Row_ST[ti+1,:],Ibeta) + TradeCost(Row_ST[ti+1,:],Ibeta,trCost,'L')
                    #openP = np.dot(Row_ST[ti,:],Ibeta) + TradeCost(Row_ST[ti,:],Ibeta,trCost,'L')
                    Ibeta[0] , Ibeta[1] = num_weight(Ibeta[0],Ibeta[1],
                                 Row_ST[ti,0],Row_ST[ti,1], 5, capital)
                    opens1payoff = -LongOrShort * Row_ST[ti,0] * Ibeta[0]
                    opens2payoff = -LongOrShort * Row_ST[ti,1] * Ibeta[1]
                    openP = tax(opens1payoff,trCost)+tax(opens2payoff,trCost)
                    Count[0,0] = Count[0,0] + 1
                    Position = 1
                    LogTradeTime[0,ti] = 1
                    opencount += 1
                    opentime = ti
            else:
                break
    Profit[0,0]=sum(Profit[0,1:5])
    trade_capital = 0
    if opens1payoff > 0 and  opens2payoff > 0:
        trade_capital = abs(opens1payoff)+abs( opens2payoff)
    elif opens1payoff > 0 and opens2payoff < 0 :
        trade_capital = abs(opens1payoff)+0.9*abs( opens2payoff)
    elif opens1payoff < 0 and opens2payoff > 0 :
        trade_capital = 0.9*abs(opens1payoff)+abs( opens2payoff)
    elif opens1payoff < 0 and opens2payoff < 0 :
        trade_capital = 0.9*abs(opens1payoff)+0.9*abs(opens2payoff)
    
    
    return [Profit, Count,opentime,closetime,trade_capital]

def trade_down(spread , trend , Row_ST ,Ibeta , trStd , trCost , tros, trcs, trMtp ):
    #spread:共整合序列 , trend:時間趨勢均值 , RowST:原始股價
    #Ibeta:整數化股票張數 , trStd:建模期間標準差 , trCost:交易成本
    #tros:開倉倍數, trcs:停損倍數
    '''    
    #Debug用
    spread, trend = cy[inNum:DailyNum+1,pi] ,cy_mean[inNum:DailyNum+1,pi]
    Row_ST= Smin[inNum:DailyNum+1,[int(OMinx[pi,0]), int(OMinx[pi,1])]] 
    Ibeta ,trStd, trCost ,tros , trcs= IntegerB[:,pi], SStd ,Cost ,Os, Fs
    trMtp = Max_tp
    '''
    #[總獲利,平倉獲利,停損獲利,換日強停獲利,換日強停虧損]
    Profit = np.zeros((1,5))
    #[開倉次數,平倉次數,停損次數,換日強停獲利次數,換日強停虧損次數]
    Count = np.zeros((1,5))
    OpenTrend = trend + trStd*tros
    StopTrend = trend + trStd*trcs
    Position = 0 # 部位控制
    LogTradeTime = np.zeros((1,spread.shape[0])) # 時間紀錄
    ForceP = 0
    openP = 0
    opencount = 0
    opentime = 0
    closetime = 0
    LongOrShort = -1
    opens1payoff = 0
    opens2payoff = 0
    closes1payoff = 0
    closes2payoff = 0
    capital = 50000000
    for ti in range(spread.shape[0]):
        
        #尾盤的強制平倉處理
        if ti == spread.shape[0]-1:
            #若有倉則強制平倉
            if Position == 1: 
                #ForceP = openP - np.dot(Row_ST[ti,:],Ibeta) + TradeCost(Row_ST[ti,:],Ibeta,trCost,'L') 
                closes1payoff = LongOrShort * Row_ST[97,0] * Ibeta[0]
                closes2payoff = LongOrShort * Row_ST[97,1] * Ibeta[1]
                ForceP = tax(closes1payoff,trCost)+tax(closes2payoff,trCost)+openP
                if ForceP > 0:
                    Profit[0,3] = ForceP
                    Count[0,3] = Count[0,3] + 1
                    Position = 0
                    LogTradeTime[0,ti] = 3
                    closetime = 97
                elif ForceP <= 0:
                    Profit[0,4] = ForceP
                    Count[0,4] = Count[0,4] + 1
                    Position = 0
                    LogTradeTime[0,ti] = 3
                    closetime = 97
        #尾盤前的交易
        else:
            if opencount <= 1 :
            #到期前若碰到平倉門檻且有倉，平倉
                if Position == 1 and spread[ti] <= trend[ti] and ti < 97:
                    #Profit[0,1] = openP - np.dot(Row_ST[ti+1,:],Ibeta) - TradeCost(Row_ST[ti+1,:],Ibeta,trCost,'L') 
                    closes1payoff = LongOrShort * Row_ST[ti,0] * Ibeta[0]
                    closes2payoff = LongOrShort * Row_ST[ti,1] * Ibeta[1]
                    ForceP = tax(closes1payoff,trCost)+tax(closes2payoff,trCost)+openP
                    #Profit[0,1] = openP - np.dot(Row_ST[ti,:],Ibeta) + TradeCost(Row_ST[ti,:],Ibeta,trCost,'L') 
                    Profit[0,1] = ForceP
                    Count[0,1] = Count[0,1] + 1
                    Position = 0 
                    LogTradeTime[0,ti] = -1
                    closetime = ti
                #到期前若碰到停損門檻且有倉，停損    
                elif Position == 1 and spread[ti]>=StopTrend[ti]:
                    #Profit[0,1] = openP - np.dot(Row_ST[ti+1,:],Ibeta) - TradeCost(Row_ST[ti+1,:],Ibeta,trCost,'L')  
                    Profit[0,1] = openP - np.dot(Row_ST[ti,:],Ibeta) + TradeCost(Row_ST[ti,:],Ibeta,trCost,'L')
                    Count[0,2] = Count[0,2] + 1
                    Position = -1
                    LogTradeTime[0,ti] = -2
                    closetime = ti
                #到期前，若碰到開倉門檻且無倉，開倉    
                elif Position == 0 and spread[ti]>=OpenTrend[ti] and ti<(trMtp-150) and opencount != 1:
                    #openP = np.dot(Row_ST[ti+1,:],Ibeta) - TradeCost(Row_ST[ti+1,:],Ibeta,trCost,'S')
                    #openP = np.dot(Row_ST[ti,:],Ibeta) - TradeCost(Row_ST[ti,:],Ibeta,trCost,'S')
                    Ibeta[0] , Ibeta[1] = num_weight(Ibeta[0],Ibeta[1],
                                 Row_ST[ti,0],Row_ST[ti,1], 5, capital)
                    opens1payoff = -LongOrShort * Row_ST[ti,0] * Ibeta[0]
                    opens2payoff = -LongOrShort * Row_ST[ti,1] * Ibeta[1]
                    openP = tax(opens1payoff,trCost)+tax(opens2payoff,trCost)
                    Count[0,0] = Count[0,0] + 1
                    Position = 1
                    LogTradeTime[0,ti] = 1
                    opencount += 1
                    opentime = ti
            else:
                break
    
    Profit[0,0]=sum(Profit[0,1:5])
    trade_capital = 0
    if opens1payoff > 0 and  opens2payoff > 0:
        trade_capital = abs(opens1payoff)+abs( opens2payoff)
    elif opens1payoff > 0 and opens2payoff < 0 :
        trade_capital = abs(opens1payoff)+0.9*abs( opens2payoff)
    elif opens1payoff < 0 and opens2payoff > 0 :
        trade_capital = 0.9*abs(opens1payoff)+abs( opens2payoff)
    elif opens1payoff < 0 and opens2payoff < 0 :
        trade_capital = 0.9*abs(opens1payoff)+0.9*abs(opens2payoff)
        
    return [Profit, Count,opentime,closetime,trade_capital]

