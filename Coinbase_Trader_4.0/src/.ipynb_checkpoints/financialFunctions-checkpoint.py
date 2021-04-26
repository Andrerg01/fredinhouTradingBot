from numba import njit
import numpy as np

@njit(nogil = True)
def returns(arrIn):
    arrOut = np.array([np.nan for _ in range(len(arrIn))])
    for i in range(1, len(arrIn)):
        arrOut[i] = (arrIn[i] - arrIn[i-1])/arrIn[i-1]
    return arrOut

@njit(nogil = True)
def logReturns(arrIn):
    arrOut = np.array([np.nan for _ in range(len(arrIn))])
    for i in range(1, len(arrIn)):
        arrOut[i] = np.log(arrIn[i]/arrIn[i-1])
    return arrOut
    
@njit(nogil = True)
def MA(arrIn, window = 60):
    arrOut = np.array([np.nan for _ in range(len(arrIn))])
    for i in range(window, len(arrIn)):
        arrOut[i] = np.mean(arrIn[i-window:i])
    return arrOut

@njit(nogil = True)
def EMA(arrIn, window = 60):
    arrOut = np.array([np.nan for _ in range(len(arrIn))])
    arrOut[0] = arrIn[0]
    k = 2./(window + 1.)
    for i in range(1, len(arrIn)):
        arrOut[i] = arrIn[i]*k + arrOut[i-1]*(1-k)
    return arrOut

@njit(nogil = True)
def MACD(arrIn, window1 = 12, window2 = 26):
    ema1 = EMA(arrIn, window = window1)
    ema2 = EMA(arrIn, window = window2)
    return ema1 - ema2

@njit(nogil = True)
def DEA(arrIn, window1 = 12, window2 = 26, window3 = 9):
    macd = MACD(arrIn, window1 = window1, window2 = window2)
    return EMA(macd, window = window3)

@njit(nogil = True)
def OSC(arrIn, window1 = 12, window2 = 26, window3 = 9):
    macd = MACD(arrIn, window1 = window1, window2 = window2)
    dea = DEA(arrIn, window1 = window1, window2 = window2, window3 = window3)    
    return macd - dea

@njit(nogil = True)
def RSI(arrIn, window = 14):
    up = np.array([np.nan for _ in range(len(arrIn))])
    down = np.array([np.nan for _ in range(len(arrIn))])
    up[0] = 0
    down[0] = 0
    for i in range(1, len(arrIn)):
        delta = arrIn[i] - arrIn[i-1]
        if delta > 0:
            up[i] = delta
            down[i] = 0
        else:
            up[i] = 0
            down[i] = -delta
    ema_up = EMA(up, window = window)
    ema_down = EMA(down, window = window)
    rs = ema_up/ema_down
    return 100. - 100./(1.+rs)

@njit(nogil = True)
def OBV(arrInClose, arrInVol):
    arrOut = np.array([np.nan for _ in range(len(arrInClose))])
    arrOut[0] = arrInVol[0]
    for i in range(1, len(arrInClose)):
        arrOut[i] = arrOut[i-1] + np.sign(arrInClose[i] - arrInClose[i-1])*arrInVol[i]
    return arrOut

@njit(nogil = True)
def nPeriodLow(arrIn, window = 7):
    arrOut = np.array([np.nan for _ in range(len(arrIn))])
    for i in range(window, len(arrIn)):
        arrOut[i] = np.min(arrIn[i-window:i])
    return arrOut

@njit(nogil = True)
def nPeriodHigh(arrIn, window = 7):
    arrOut = np.array([np.nan for _ in range(len(arrIn))])
    for i in range(window, len(arrIn)):
        arrOut[i] = np.max(arrIn[i-window:i])
    return arrOut

@njit(nogil = True)
def backtestStrategyA(arrClose, MAperiod, nDayLow, nDayHigh, nLimits):
    arrMA = MA(arrClose, window = MAperiod)
    arrLow = nPeriodLow(arrClose, window = nDayLow)
    arrHigh = nPeriodHigh(arrClose, window = nDayHigh)
    purchasedQ = False
    arrPurchased = np.array([False for _ in range(len(arrClose))])
    arrReturn = np.array([np.nan for _ in range(len(arrClose))])
    arrLength = np.array([np.nan for _ in range(len(arrClose))])
    for i in range(len(arrClose)):
        if not purchasedQ and arrClose[i] < arrLow[i] and arrClose[i] > arrMA[i]:
            purchasedQ = True
            priceIn = arrClose[i]
            indexIn = i
        elif purchasedQ and arrClose[i] > arrHigh[i]:
            purchasedQ = False
            arrReturn[i] = (arrClose[i] - priceIn)/priceIn
            arrLength[i] = i - indexIn
        arrPurchased[i] = purchasedQ
    arrReturn = arrReturn[~np.isnan(arrReturn)]
    arrLength = arrLength[~np.isnan(arrLength)]

    if arrReturn.shape[0] < nLimits[0] or arrReturn.shape[0] > nLimits[1]:
        score = 0
    else:
        score = np.mean(arrReturn)/np.std(arrReturn)
    
    return arrReturn, arrLength, arrPurchased, score, len(arrReturn), np.array([MAperiod, nDayLow, nDayHigh])

@njit(nogil = True)
def backtestStrategyB(arrClose, RSIperiod, RSILow, RSIHigh, nLimits):
    arrRSI = RSI(arrClose, window = RSIperiod)
    purchasedQ = False
    arrPurchased = np.array([False for _ in range(len(arrClose))])
    arrReturn = np.array([np.nan for _ in range(len(arrClose))])
    arrLength = np.array([np.nan for _ in range(len(arrClose))])
    for i in range(1, len(arrClose)):
        if not purchasedQ and arrRSI[i-1] < RSIHigh and arrRSI[i] >= RSIHigh:
            purchasedQ = True
            priceIn = arrClose[i]
            indexIn = i
        elif purchasedQ and arrRSI[i-1] > RSILow and arrRSI[i] <= RSILow:
            purchasedQ = False
            arrReturn[i] = (arrClose[i] - priceIn)/priceIn
            arrLength[i] = i - indexIn
        arrPurchased[i] = purchasedQ

    arrReturn = arrReturn[~np.isnan(arrReturn)]
    arrLength = arrLength[~np.isnan(arrLength)]
    
    if arrReturn.shape[0] < nLimits[0] or arrReturn.shape[0] > nLimits[1]:
        score = 0
    else:
        score = np.mean(arrReturn)/np.std(arrReturn)
        
    return arrReturn, arrLength, arrPurchased, score, len(arrReturn), np.array([RSIperiod, RSILow, RSIHigh])

@njit(nogil = True)
def backtestStrategyC(arrClose, OSCPeriod1, OSCPeriod2, OSCPeriod3, OSCLow, OSCHigh, nLimits):
    arrOSC = OSC(arrClose, window1 = OSCPeriod1, window2 = OSCPeriod2, window3 = OSCPeriod3)
    arrOSC = arrOSC/arrClose
    purchasedQ = False
    arrPurchased = np.array([False for _ in range(len(arrClose))])
    arrReturn = np.array([np.nan for _ in range(len(arrClose))])
    arrLength = np.array([np.nan for _ in range(len(arrClose))])
    for i in range(1, len(arrClose)):
        if not purchasedQ and arrOSC[i-1] < OSCHigh and arrOSC[i] >= OSCHigh:
            purchasedQ = True
            priceIn = arrClose[i]
            indexIn = i
        elif purchasedQ and arrOSC[i-1] > OSCLow and arrOSC[i+1] <= OSCLow:
            purchasedQ = False
            arrReturn[i] = (arrClose[i] - priceIn)/priceIn
            arrLength[i] = i - indexIn
        arrPurchased[i] = purchasedQ

    arrReturn = arrReturn[~np.isnan(arrReturn)]
    arrLength = arrLength[~np.isnan(arrLength)]
    
    if arrReturn.shape[0] < nLimits[0] or arrReturn.shape[0] > nLimits[1]:
        score = 0
    else:
        score = np.mean(arrReturn)/np.std(arrReturn)
    return arrReturn, arrLength, arrPurchased, score, len(arrReturn), np.array([OSCPeriod1, OSCPeriod2, OSCPeriod3, OSCLow, OSCHigh])

def nicefyPortfolio(portfolio, treshold):
    #Zeroes entries below 'treshold' and normalizes portfolio (dict)
    port = normalizePortfolio(portfolio)
    
    for key in port.keys():
        if port[key] < treshold:
            port[key] = 0

    port = normalizePortfolio(port)
    return port

def normalizePortfolio(portfolio):
    #normalized portfolio (dict)
    total = sum([portfolio[key] for key in portfolio.keys()])
    if total == 0:
        total = 1
    for key in portfolio.keys():
        portfolio[key] = portfolio[key]/total
    return portfolio

def printPortfolio(portfolio):
    strOut = ""
    for key in portfolio.keys():
        if portfolio[key] > 0:
            strOut += key + ":" + str(portfolio[key]) + " | "
    return strOut[:-3]

def nicefyTrades(buys, sells, treshold = 0.05):
    #Necefies sales dict by redistributing the ones below treshold
    lowKeysSells = []
    highKeysSells = []
    lowKeysBuys = []
    highKeysBuys = []
    
    for key in sells.keys():
        if sells[key] < treshold:
            lowKeysSells += [key]
        else:
            highKeysSells += [key]
    for key in buys.keys():
        if buys[key] < treshold:
            lowKeysBuys += [key]
        else:
            highKeysBuys += [key]
            
    for lowKey in lowKeysSells:
        for key in highKeysSells:
            sells[key] += sells[lowKey]/len(highKeysSells)
        sells[lowKey] = 0
    for lowKey in lowKeysBuys:
        for key in highKeysBuys:
            buys[key] += buys[lowKey]/len(highKeysBuys)
        buys[lowKey] = 0
    tempBuys = {}
    for key in buys.keys():
        if buys[key] > 0:
            tempBuys[key] = buys[key]
    tempSells = {}
    for key in sells.keys():
        if sells[key] > 0:
            tempSells[key] = sells[key]
    buys = tempBuys.copy()
    sells = tempSells.copy()
    
    if len(buys) == 0 or len(sells) == 0:
        return {}, {}
    
    totalBuys = sum([buys[key] for key in buys.keys()])
    totalSells = sum([sells[key] for key in sells.keys()])
    
    newTotal = (totalBuys + totalSells)/2
    for key in buys.keys():
        buys[key] = buys[key]*newTotal/totalBuys
    for key in sells.keys():
        sells[key] = sells[key]*newTotal/totalSells
        
    return buys, sells

def makePricePlot(hist, ax):
    ax.plot(hist['Funds'], label = 'Total Funds', color = 'green')
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Funds (USD)")
    ax.set_title("Funds History")
    ax.grid()
    
def makeTimePortfolioPlot(hist, ax):
    everInvestedAssets = []
    for port in hist['Portfolio']:
        for key in port.keys():
            if port[key] > 0 and key not in everInvestedAssets:
                everInvestedAssets += [key]
    assetInvestOverTime = {asset:[] for asset in everInvestedAssets}
    for port in hist['Portfolio']:
        for asset in everInvestedAssets:
            if asset in port.keys():
                assetInvestOverTime[asset] += [port[asset]]
            else:
                assetInvestOverTime[asset] += [0]
    for asset in everInvestedAssets:
        ax.plot(hist.index, assetInvestOverTime[asset], label = asset)
    ax.legend(title='Asset', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title('Asset share in portfolio over time.')
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio share")
    ax.grid()
    
def makeTimePortfolioPlot(hist, ax):
    boxProps = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    everInvestedAssets = []
    for port in hist['Portfolio']:
        for key in port.keys():
            if port[key] > 0 and key not in everInvestedAssets:
                everInvestedAssets += [key]
    assetInvestOverTime = {asset:[] for asset in everInvestedAssets}
    for port in hist['Portfolio']:
        for asset in everInvestedAssets:
            if asset in port.keys():
                assetInvestOverTime[asset] += [port[asset]]
            else:
                assetInvestOverTime[asset] += [0]
    for i in range(len(everInvestedAssets)):
        if i == 0:
            y1 = assetInvestOverTime[everInvestedAssets[i]]
            ax.fill_between(hist.index, y1)
            if assetInvestOverTime[everInvestedAssets[i]][-1] > 0.01:
                x = hist.index[-1]
                y = assetInvestOverTime[everInvestedAssets[i]][-1]/2
                ax.text(x, y, everInvestedAssets[i], bbox = boxProps)
        else:
            y1 = assetInvestOverTime[everInvestedAssets[i]] + sum([np.array(assetInvestOverTime[everInvestedAssets[j]]) for j in range(i)])
            y2 = sum([np.array(assetInvestOverTime[everInvestedAssets[j]]) for j in range(i)])
            ax.fill_between(hist.index, y1, y2, where = y1 > y2, interpolate = True)
            if assetInvestOverTime[everInvestedAssets[i]][-1] > 0.01:
                x = hist.index[-1]
                y = assetInvestOverTime[everInvestedAssets[i]][-1]/2 + sum([np.array(assetInvestOverTime[everInvestedAssets[j]]) for j in range(i)])[-1]
                ax.text(x, y, everInvestedAssets[i], bbox = boxProps)
    ax.set_title('Asset share in portfolio over time.')
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio share")
    ax.grid() 

    
def makeCurrentPortfolioPlot(hist, ax):
    everInvestedAssets = []
    for port in hist['Portfolio']:
        for key in port.keys():
            if port[key] > 0 and key not in everInvestedAssets:
                everInvestedAssets += [key]
    heights = [hist.iloc[-1]['Portfolio'][asset] for asset in hist.iloc[-1]['Portfolio'].keys()]
    ax.bar([asset[:3] for asset in hist.iloc[-1]['Portfolio'].keys()], heights)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.set_xlabel("Asset")
    ax.set_ylabel("Share in current portfolio")
    ax.set_title("Layout of current portfolio")
    ax.grid()

def makeMarketPerformancePlot(allCloseRelevant, ax):
    equalWeightFunds = [1 for _ in range(len(allCloseRelevant[0]))]
    
    equalWeightReturns = [0 for _ in range(len(allCloseRelevant[0]))]
    for i in range(len(allCloseRelevant)):
        for j in range(1, len(allCloseRelevant[i])):
            equalWeightReturns[j] = (allCloseRelevant[i][j] - allCloseRelevant[i][j-1])/allCloseRelevant[i][j-1]/len(allCloseRelevant)
            equalWeightFunds[j] = equalWeightFunds[j-1]*(1+equalWeightReturns[j])
    ax.plot(equalWeightFunds, '--', color = 'blue', label = 'Equally Weighted Backtest')    
    ax.set_title("Market Performance")
    ax.set_ylabel("Equally Weighted Portfolio Funds")
    ax.set_xlabel("Periods")
    ax.legend(loc='upper left')
    ax.grid()

    
