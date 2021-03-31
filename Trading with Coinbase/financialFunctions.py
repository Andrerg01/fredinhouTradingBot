import cbpro
import numpy as np
import pandas as pd
import pickle as pkl
import datetime
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from IPython.display import clear_output
import os
from decimal import *

def clear():
    try:
        get_ipython().__class__.__name__
        clear_output()
    except:
        os.system('clear')
        
class TextWebsocketClient(cbpro.WebsocketClient):
    #useless crap
    def on_open(self):
        self.url = 'wss://ws-feed-public.sandbox.pro.coinbase.com'
        self.message_count = 0
        pass
    
    def on_message(self, msg):
        self.message_count += 1
        msg_type = msg.get('type',None)
        if msg_type == 'ticker':
            time_val   = msg.get('time',('-'*27))
            price_val  = msg.get('price',None)
            if price_val is not None:
                price_val = float(price_val)
            product_id = msg.get('product_id',None)
            
            print(f"{time_val:30} \
                {price_val:.3f} \
                {product_id}\tchannel type:{msg_type}")
    
    def on_close(self):
        print(f"<---Websocket connection closed--->\n\tTotal messages: {self.message_count}")
        
def progressBar(pct, time0 = 0, header = ''):
    time1 = datetime.datetime.now()
    if time0 == 0:
        eta = ''
    else:
        deltaT = (time1-time0).total_seconds()
        try:
            eta = 100.*deltaT/pct - deltaT
        except:
            eta = ''
    numberOfHashtags = int(pct/2) + 1
    numberOfdots = 50-numberOfHashtags
    outString = "[" + "#"*numberOfHashtags + '.'*numberOfdots + '] '
    if eta != '':
        #outString += '(approx ' + '%0.2f' % eta + ' seconds)'
        if eta/(60*60*24) > 1:
            outString += '(approx > 1d)'
        elif eta/(60*60) > 1:
            outString += '(approx ' + time.strftime('%Hh %Mm %Ss', time.gmtime(eta)) +  ')'
        elif eta/60 > 0:
            outString += '(approx ' + time.strftime('%Mm %Ss', time.gmtime(eta)) +  ')'
        else:
            outString += '(approx ' + time.strftime('%Ss', time.gmtime(eta)) +  ')'
    return header + outString

def getCandles(Client, asset, start, end, granularity, triesMax = 10, verbose = True):
    #Function to get candles data straight from coinbase for a single asset
    #Keep track of thenumber of attepts, maximum = 10 by default
    tries = 0
    #Keeps track if the candles have been sucessfully imported
    success = False
    #Loops for 10 tries or until we get the data successfully.
    while tries < triesMax and not success:
        #standard function to call for historic data
        candles = Client.get_product_historic_rates(asset, start = start.isoformat(), end = end.isoformat(), granularity = granularity)
        #candles will be 1 unit long if it is an error message
        if len(candles) != 1:
            success = True
        else:
            if verbose:
                print("Failure in retreiving data for " + asset + ". Waiting 1 second and trying again. " + str(tries + 1) + "/" + str(triesMax))
            tries += 1
            time.sleep(.1)
    if not success:
        print("Error getting candles for " + asset)
        return []
    else:
        return candles
            

def getRawData(Client, asset, start = None, end = None, granularity = None, verbose = True, header = ''):
    """
    Returns a pandas DF with the 'time', 'low', 'high', 'open', 'close', 'volume', values for a given asset in the stated period and the given granularity.
    ('time' is the imdex).    
    """
    
    allowedGranularity = [60, 300, 900, 3600, 21600, 86400]
    if end == None:
        end = datetime.datetime.now()
    if start == None:
        start = end - datetime.timedelta(days = 30)
    if granularity == None:
        granularity = 21600
    if granularity not in allowedGranularity:
        print("Error: Granularity not one of the allowed values")
        return
    
    #Variable to keep track of existance of previous data file
    rawDataFile = False
    try:
        #If file is opened successfully, cool, file variable is real.
        with open("RawData/" + asset + "_RawData.pkl", 'rb') as f:
            rawDataFile = True
    except:
        #If error, then file variable is false
        rawDataFile = False
    if rawDataFile:
        #If there is a file, open it and load into variable
        with open("RawData/" + asset + "_RawData.pkl", 'rb') as f:
            candles_df = pkl.load(f)
        #This puts the index back as a column
        candles_df.reset_index(inplace = True)
        #Turns into array like the ones from the getCandles method
        candlesTemp = candles_df.to_numpy().tolist()
        candles0 = []
        #Saves all the candles with timestamp after "start" and discards the rest
        for candle in candlesTemp:
            if candle[0].to_pydatetime() >= start:
                candles0 += [[candle[0].to_pydatetime()] + candle[1:]]
        #New start is the last time stamp checked + granularity
        start = max([candle[0] for candle in candles0])
    else:
        candles0 = []
    #checks if more then 300 indices will be imported, if so, it breaks it down into smaller pieces and puts it together after. Coinbase only allows up to 300
    if (end - start).total_seconds()/granularity > 300:
        if verbose:
            print("Retrieving data for " + asset)
            print("Granularity too small, subdividing requests.")
        newStart = start
        newEnd = start + datetime.timedelta(seconds = 300*granularity)
        candles = []
        i = 1
        iMax = int((end - start).total_seconds()/(300*granularity))+1
        while newEnd < end:
            if verbose:
                clear()
                print(progressBar(i/iMax*100, header = header + '\n'))
            candles += getCandles(Client, asset, newStart, newEnd, granularity, verbose = verbose)
            newStart = newEnd
            newEnd = newEnd + datetime.timedelta(seconds = 300*granularity)
            i += 1
        candles += getCandles(Client, asset, newEnd - datetime.timedelta(seconds = 300*granularity), end, granularity, verbose = verbose)
    else:
        clear()
        print(header)
        candles = getCandles(Client, asset, start, end, granularity, verbose = verbose)
    for i in range(len(candles)):
        #Converts time index
        time_in_dt = datetime.datetime.fromtimestamp(candles[i][0])
        candles[i][0] = time_in_dt
    #Prepends the already-collected data
    candles = candles0 + candles
    
    cTimes = []
    cTemp = []
    for row in candles:
        if row[0] not in cTimes:
            cTemp += [row]
            cTimes += [row[0]]
    candles = cTemp.copy()
    
    candles_DF = pd.DataFrame(candles, columns = ['time', 'low', 'high', 'open', 'close', 'volume']).set_index('time').sort_index().dropna()
    
    #Saves new data into file
    with open("RawData/" + asset + "_RawData.pkl", 'wb') as f:
        pkl.dump(candles_DF, f)
    return candles_DF

def getClosingPrice(Client, candles_df = None, asset = None, start = None, end = None, granularity = None, verbose = True, usd = False, header = ''):
    #if candles_df is provided, it calculates from there, if not, it imports necessary data. Supports assets to be strings or lists
    if isNone(candles_df):
        if asset == None:
            print("Error, no asset or candles specified when calling getClosingPrice(). Please specify at least one.")
            return
        if isinstance(asset, str):
            candles_df = getRawData(Client, asset, start = start, end = end, granularity = granularity, verbose = verbose)
            closingPrice_df = pd.DataFrame(candles_df['close'])
            closingPrice_df.columns = [asset]
        else:
            cp = []
            time0 = datetime.datetime.now()
            for i in range(len(asset)):
                clear()
                headerTemp = header + "\nGathering data for " + str(asset[i]) + "\n" + progressBar(i/len(asset)*100, time0 = time0) + "\n"
                cp += [getRawData(Client, asset[i], start = start, end = end, granularity = granularity, verbose = verbose, header = headerTemp)['close']]
            #Removes repeated indexes
            closingPrice_df = pd.DataFrame(cp)
            closingPrice_df = closingPrice_df.T
            closingPrice_df.columns = asset
    else:
        closingPrice_df = pd.DataFrame(candles_df['close'])
        closingPrice_df.columns = [asset]
    #Option to add USD as a columns of 1's
    if usd:
        usdVals = [1 for i in range(len(closingPrice_df))]
        closingPrice_df['USD-USD'] = usdVals
    return closingPrice_df.dropna()

def getFunds(Client, price_df, coin = 'funds'):
    #Returns value invested in each asset. buy is in t
    accs = Client.get_accounts()
    assets = list(price_df.columns)
    funds = {asset:0 for asset in assets}
    for acc in accs:
        if acc['currency'] + "-USD" in assets:
            if coin == 'funds':
                funds[acc['currency'] + "-USD"] = eval(acc['balance'])*price_df[acc['currency'] + "-USD"][-1]
            elif coin == 'size':
                funds[acc['currency'] + "-USD"] = eval(acc['balance'])
    return funds

def getPortfolio(Client, price_df):
    funds = getFunds(Client, price_df)
    totalFunds = sum([funds[key] for key in funds.keys()])
    if totalFunds == 0:
        totalFunds = 1
    portfolio = {key:funds[key]/totalFunds for key in funds.keys()}
    return portfolio

def isNone(var):
    #Check if a variable == None easily
    try:
        len(var)
        return False
    except:
        return True
        
def getReturns(Client, price_df = None, candles_df = None, asset = None, start = None, end = None, granularity = None, verbose = True, usd = False):
    #Gets the returns series for given asset(s)
    if isNone(price_df):
        print('Should not be here')
        closing_df = getClosingPrice(Client, candles_df = candles_df, asset = asset, start = start, end = end, granularity = granularity, verbose = verbose, usd = usd)  
    returns_df = price_df.pct_change().dropna()
    return returns_df
    
def periodsPerInterval(returns_df, interval = 'W'):
    #How many periods of data fit in a given time interval
    seconds_per_period = (returns_df.index[1] - returns_df.index[0]).total_seconds()
    seconds_per_interval = pd.to_timedelta(1, unit = interval).total_seconds()
    return seconds_per_interval/seconds_per_period

def intervalizeReturns(returns_df, interval):
    #Like annualized, but for any interval. Not that necessary now that I think about it but cool for weekly porjections or whatever
    compoundedGrowth = (1+returns_df).prod()
    nPeriods = len(returns_df)
    return compoundedGrowth**(periodsPerInterval(returns_df, interval = interval)/nPeriods) - 1

def intervalizeVolatility(returns_df, interval = 'W'):
    #Same as intervalizeReturns but for volatilities
    return returns_df.std(ddof = 0)*(periodsPerInterval(returns_df, interval = 'W')**0.5)

def sharpe_ratio(returns_df, interval = 'W', riskfree_rate = 0.03):
    #Calculates sharpe ratio, look it up
    riskfree_per_period = (1+riskfree_rate)**(1/periodsPerInterval(returns_df, interval = interval)) - 1
    excess_return = returns_df - riskfree_per_period
    intervalized_excess_return = intervalizeReturns(excess_return, interval = interval)
    intervalized_volatility = intervalizeVolatility(returns_df)
    return intervalized_excess_return/intervalized_volatility

def portfolioReturn(returns_df, weights, interval = 'W'):
    #Calculates the return for a given portfolio (given as returns_df and weights only)
    return weights.T@intervalizeReturns(returns_df, interval)

def portfolioVolatility(returns_df, weights):
    #Calculates the volatility for a given portfolio (given as returns_df and weights only)
    return (weights.T@returns_df.cov()@weights)**(0.5)

def portRetTemp(weights, returns_df, interval):
    #Same as portfolioReturn but different prder of arguments for minimization
    return portfolioReturn(returns_df, weights, interval)
def portVolTemp(weights, returns_df):
    #Same as portfolioVolatility but different prder of arguments for minimization
    return portfolioVolatility(returns_df, weights)

def minimizeVolatility(target_return, returns_df, interval = 'W'):
    #Finds the best set of weights which minimize the volatility for a given return
    """
    target_return -> W
    """
    n = returns_df.T.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    return_is_target = {
        'type': 'eq',
        'args': (returns_df,interval,),
        'fun': lambda weights, returns_df, interval: target_return - portRetTemp(weights, returns_df, interval)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(portVolTemp, init_guess, 
                       args = (returns_df,),  method = "SLSQP", 
                       options = {'disp': False},
                       constraints = (return_is_target, weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x

def maximumSharpeRatio(riskfreeRate, returns_df, interval = 'W'):
    #Finds the weights of the portfolio that minimizes the sharpe ratio
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio given
    the riskfree rate and expected returns and a covariance matrix
    """
    er = intervalizeReturns(returns_df, interval = interval)
    cov = returns_df.cov()
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    def negSharpeRatio(weights, riskfreeRate, er, cov):
        """
        Returns the negative of the Sharpe Ratio, given weights
        """
        r = portfolioReturn(returns_df, weights, interval = interval)
        vol = portfolioVolatility(returns_df, weights)
        return -(r - riskfreeRate)/vol
    results = minimize(negSharpeRatio, init_guess, 
                       args = (riskfreeRate,er,cov,),  method = "SLSQP", 
                       options = {'disp': False},
                       constraints = (weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x

def nicefyPortfolio(portfolio, treshold):
    #Zeroes entries below 'treshold' and normalizes portfolio (dict)
    port = normalizePortfolio(portfolio)
    
    for key in port.keys():
        if port[key] < treshold:
            port[key] = 0
            
    portSorted = {}
    keys = sorted(port)
    for key in keys:
        portSorted[key] = port[key]
        
    port = portSorted.copy()
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

def makePortfolio(availableAssets, consideredAssets, weights, treshold = 0.05):
    #Makes a portfolio dict from given assets and weights then nicefies it
    portfolio = {asset:0 for asset in availableAssets}
    for i in range(len(consideredAssets)):
        portfolio[consideredAssets[i]] = weights[i]
    return nicefyPortfolio(portfolio, treshold = treshold)

def portfoliosDistance(returns_df, pfA, pfB, xRange, yRange, interval = 'W'):
    #Calculates the relative distance between portfolios (dict)
    wA = []
    wB = []
    for key in pfA.keys():
        wA += [pfA[key]]
        wB += [pfB[key]]
    wA = np.array(wA)
    wB = np.array(wB)
    
    XA = portfolioVolatility(returns_df, wA)/xRange
    YA = portfolioReturn(returns_df, wA, interval = 'W')/yRange
    
    XB = portfolioVolatility(returns_df, wB)/xRange
    YB = portfolioReturn(returns_df, wB, interval = 'W')/yRange
    return np.sqrt((XA-XB)**2 + (YA-YB)**2)

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

def makeTrades(Client, buys, sells):
    for key in sells.keys():
        if key != 'USD-USD':
            placed = False
            completed = False
            pct = 1.0
            size = sells[key]*pct
            while not placed and pct > 0:
                print("Placing an order to sell " + str(size) + " " + key)
                sellOrder = Client.place_market_order(product_id = key, side = 'sell', size = size)
                if len(sellOrder) > 1:
                    placed = True
                    time.sleep(1)
                else:
                    print(sellOrder)
                    if sellOrder['message'].startswith("size is too accurate. Smallest unit is"):
                        minAmmount = sellOrder['message'].split(" ")[-1]
                        if minAmmount[0] == '1':
                            accuracy = 0
                        else:
                            accuracy = len(sellOrder['message'].split(" ")[-1].split(".")[-1].split('1')[0]) + 1
                        size = round(size, accuracy)
                        print("Order amount too precise, changing size accordingly")
                    elif sellOrder['message'].startswith("size is too small."):
                        pct = 0
                        print("Too tired for this crap")
                    else:    
                        pct -= 0.005
                        size = sells[key]*pct
                        print("Order placement not successfull, trying again with " + str(pct*100) + "% of original amount in 1 second")
            if placed:
                while not completed:
                    orderStatus = Client.get_order(sellOrder['id'])['status']
                    if orderStatus == 'done':
                        print("Order to sell " + str(size) + " " + key + " has been successfully completed.")
                        completed = True
                    else:
                        print("Order did not yet complete, waiting 1 second and trying again.")
                        time.sleep(1)  
    for key in buys.keys():
        if key != 'USD-USD':
            placed = False
            completed = False
            pct = 1.0
            size = buys[key]*pct
            while not placed and pct > 0:
                print("Placing an order to buy " + str(size) + " " + key)
                buyOrder = Client.place_market_order(product_id = key, side = 'buy', size = size)
                if len(buyOrder) > 1:
                    placed = True
                    time.sleep(1)
                else:
                    print(buyOrder)
                    if buyOrder['message'].startswith("size is too accurate. Smallest unit is"):
                        minAmmount = buyOrder['message'].split(" ")[-1]
                        if minAmmount[0] == '1':
                            accuracy = 0
                        else:
                            accuracy = len(minAmmount.split(".")[-1].split('1')[0]) + 1
                        size = round(size, accuracy)
                        print("Order amount too precise, changing size accordingly")
                    elif buyOrder['message'].startswith("size is too small."):
                        pct = 0
                    else:    
                        pct -= 0.005
                        size = buys[key]*pct
                        print("Order placement not successfull, trying again with " + str(pct*100) + "% of original amount in 1 second")
            if placed:
                while not completed:
                    orderStatus = Client.get_order(buyOrder['id'])['status']
                    if orderStatus == 'done':
                        print("Order to buy " + str(size) + " " + key + " has been successfully completed.")
                        completed = True
                    else:
                        print("Order did not yet complete, waiting 1 second and trying again.")
                        print(orderStatus)
                        time.sleep(1)                 
            