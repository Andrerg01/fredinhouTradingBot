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
from os import listdir
from os.path import isfile, join

def concatReturns(data_in, dropna = True):
    """
    Concatenates a column "Returns" to data_in and returns a version of data_in with the concatenated column.
    """
    data = data_in.copy()
    if isinstance(data, list):
        for i in range(len(data)):
            ret = data[i]['Close'].pct_change()
            if dropna:
                ret = ret.dropna()
            data[i]['Return'] = ret
            if i > 0:
                data[i]['Return'].iloc[0] = (data[i].iloc[0]['Close'] - data[i-1].iloc[-1]['Close'])/data[i-1].iloc[-1]['Close']
    else:
        ret = data['Close'].pct_change()
        if dropna:
            ret = ret.dropna()
        data['Return'] = ret
    return data
   
def periodsPerInterval(dataDict, interval = 'W'):
    for key in dataDict.keys():
        asset = key
        break
    
    #How many periods of data fit in a given time interval
    seconds_per_period = (dataDict[asset].index[1] - dataDict[asset].index[0]).total_seconds()
    seconds_per_interval = pd.to_timedelta(1, unit = interval).total_seconds()
    return seconds_per_interval/seconds_per_period

def intervalizeReturns(dataDict, interval = 'W'):
    #Like annualized, but for any interval. Not that necessary now that I think about it but cool for weekly porjections or whatever
    intRets = {}
    for asset in dataDict.keys():
        compoundedGrowth = (1+dataDict[asset]['Return']).prod()
        nPeriods = len(dataDict[asset])
        intRets[asset] = compoundedGrowth**(periodsPerInterval(dataDict, interval = interval)/nPeriods) - 1
    return intRets

def intervalizeVolatility(dataDict, interval = 'W'):
    #Same as intervalizeReturns but for volatilities
    intVols = {}
    for asset in dataDict.keys():
        intVols[asset] = dataDict[asset]['Return'].std(ddof = 0)*(periodsPerInterval(dataDict, interval = interval)**0.5)
    return intVols

def sharpeRatio(dataDict, interval = 'W', riskfree_rate = 0):
    dataDictTemp = dataDict.copy()
    #Calculates sharpe ratio, look it up
    riskfree_per_period = (1+riskfree_rate)**(1/periodsPerInterval(dataDict, interval = interval)) - 1
    for key in dataDictTemp.keys():
        dataDictTemp[key]['Return'] = dataDict[key]['Return'] - riskfree_per_period
    intervalized_excess_return = intervalizeReturns(dataDictTemp, interval = interval)
    intervalized_volatility = intervalizeVolatility(dataDictTemp, interval = interval)
    return {key:intervalized_excess_return[key]/intervalized_volatility[key] for key in intervalized_excess_return.keys()}

def portfolioReturn(dataDict, portfolio, interval = 'W'):
    #Calculates the return for a given portfolio (given as dataDict and portfolio only).
    weights = []
    intReturns = []
    for key in portfolio.keys():
        weights += [portfolio[key]]
        intReturns += [intervalizeReturns({key:dataDict[key]}, interval = interval)[key]]
    weights = np.array(weights)
    intReturns = np.array(intReturns)
    return weights.T@intReturns

def portfolioVolatility(dataDict, portfolio):
    #Calculates the return for a given portfolio (given as dataDict and portfolio only).
    returns_df = pd.DataFrame({key: dataDict[key]['Return'] for key in dataDict.keys()}).dropna()
    weights = []
    intReturns = []
    for key in portfolio.keys():
        weights += [portfolio[key]]
    weights = np.array(weights)
    return (weights.T@returns_df.cov()@weights)**(0.5)

def portfolioSharpeRatio(dataDict, portfolio, interval = 'W', riskfree_rate = 0):
    r = portfolioReturn(dataDict, portfolio, interval = interval)
    v = portfolioVolatility(dataDict, portfolio)
    return (r-riskfree_rate)/v

def portfolioLogSharpeRatio(dataDict, portfolio, interval = 'W', riskfree_rate = 0):
    r = portfolioReturn(dataDict, portfolio, interval = interval)
    v = portfolioVolatility(dataDict, portfolio)
    return r-riskfree_rate/v

def maximizeStatistic(Statistic, params):
    #Finds the weights of the portfolio that minimizes the sharpe ratio
    n = params['numberOfWeights']
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    def minStatistic(weights, Statistic, params):
        dataDict = params['dataDict']
        portfolio = {}
        i = 0
        for key in dataDict.keys():
            portfolio[key] = weights[i]
            i += 1
        params['portfolio'] = portfolio
        return -Statistic(params)
            
    results = minimize(minStatistic, init_guess, 
                       args = (Statistic, params),  method = "SLSQP", 
                       options = {'disp': False},
                       constraints = (weights_sum_to_1),
                       bounds = bounds
                      )
    
    portOut = {}
    dataDict = params['dataDict']
    i = 0
    for key in dataDict.keys():
        portOut[key] = results.x[i]
        i += 1
        
    return portOut

def modifiedSigmoid(x):
    return 1/(1+np.exp(-x)) + 1/2

def myLog(x):
    return np.log(modifiedSigmoid(x))
    

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
def availableData(asset, date):
    path = '../candlesDataBase/' + asset
    starts = []
    ends = []
    for file in listdir(path):
        if file[-3:] == 'pkl':
            starts += [int(file.split("_")[1])]
            ends += [int(file.split("_")[2])]
    if max(ends) >= datetime.datetime.timestamp(date) and datetime.datetime.timestamp(date) >= min(starts):
        return True
    else:
        return False