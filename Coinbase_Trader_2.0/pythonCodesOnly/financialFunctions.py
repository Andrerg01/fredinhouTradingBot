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
import utilityFunctions as uf

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
    keys = []
    returnsDict = {}
    for key in portfolio.keys():
        returnsDict[key] = dataDict[key]['Return']
    returns_df = pd.DataFrame(returnsDict)
    #returns_df = pd.DataFrame([dataDict[key]['Return'].values for key in portfolio.keys()], columns = ['Return'], index = dataDict[keys[0]].index)
    
    weights = []
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
    return (r-riskfree_rate)/v

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
         
def availableData(asset, date):
    path = '../candlesDataBase/' + asset
    starts = []
    ends = []
    for file in listdir(path):
        if file[-3:] == 'csv':
            starts += [int(file.split("_")[1])]
            ends += [int(file.split("_")[2])]
    if max(ends) >= datetime.datetime.timestamp(date) and datetime.datetime.timestamp(date) >= min(starts):
        return True
    else:
        return False
    
def nicefyData(dataDict, granularity = 60, header = '', verbose = True):
    starts = []
    ends = []
    for key in dataDict.keys():
        starts += [list(dataDict[key].index)[0]]
        ends += [list(dataDict[key].index)[-1]]
        
    startTemp = max(starts)
    endsTemp = max(ends)
    supposedIndexes = []
    while startTemp <= max(ends):
        supposedIndexes += [startTemp]
        startTemp += datetime.timedelta(seconds = granularity)
    i = 0
    time0 = datetime.datetime.now()
    for key in dataDict.keys():
        if key != 'USD-USD':
            pct = i/len(dataDict)*100
            uf.clear()
            if verbose:
                uf.clear()
                print(uf.progressBar(pct, time0 = time0, header = header))
            shoveIdx = list(set(supposedIndexes) - set(dataDict[key].index))
            vals = [[float('nan') for j in range(len(dataDict[key].columns))] for i in range(len(shoveIdx))]
            dataDict[key] = dataDict[key].append(pd.DataFrame(vals, columns = dataDict[key].columns, index = shoveIdx)).sort_index()
            dataDict[key] = dataDict[key][min(supposedIndexes):max(supposedIndexes)].interpolate()
            i += 1
    for key in dataDict.keys():
        dataDict[key] = dataDict[key][max(starts):max(ends)] 
        
    return dataDict
    