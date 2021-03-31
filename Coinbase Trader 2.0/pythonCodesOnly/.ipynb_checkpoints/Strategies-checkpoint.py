import numpy as np
import pandas as pd
import financialFunctions as ff
import utilityFunctions as uf
import coinbaseFunctions as cf
import datetime
from scipy.optimize import minimize
#File holding the definition of all strategies.
#All strategies take in dataDict, which is a dictionary of all the assets to be considered, each entry is a dataframe with index 'Date' and columns ['Low', 'High', 'Open', 'Close', 'Volume', 'Returns'], as well as a dictionary variable called 'params', which contains the expected parameters. And they all return a list of buys and sells to be made.

def maxSharpeRatio(params):
    def Statistic(params):
        dataDict = params['dataDict']
        portfolio = params['portfolio']
        return ff.portfolioSharpeRatio(dataDict, portfolio)
    
    #Dictionary of all the 
    dataDict = params['dataDict']
    #Time which the data will look back from the last entry and calculate the return and volatility (seconds)
    lookbackTime = params['lookBackTime']
    #Smallest amount in dollars a trade is allowed to happen in.
    minimumTrade = params['minimumTrade']
    #The current portfolio when this was called.
    currentPortfolio = params['currentPortfolio']
    #The total ammount of funds available (in USD)
    totalFunds = params['totalFunds']
    #Setting number of weights for the portfolio calculation
    params['numberOfWeights'] = len(dataDict)
    
    #List of assets is the indexes of dataDict
    for key in dataDict:
        dataDict[key] = dataDict[key][max(dataDict[key].index) - datetime.timedelta(seconds = lookbackTime):]
    
    #msrPort = ff.maximumSharpeRatio(dataDict.copy())
    msrPort = ff.maximizeStatistic(Statistic, params)
    
    msrPort = ff.nicefyPortfolio(msrPort, treshold = minimumTrade/totalFunds)
    
    porA = currentPortfolio.copy()
    porB = msrPort.copy()
    
    buys = {}
    sells = {}
    for key in porA.keys():
        if porB[key] > porA[key]:
            buys[key] = porB[key] - porA[key]
        elif porB[key] < porA[key]:
            sells[key] = porA[key] - porB[key]
            
    buys, sells = ff.nicefyTrades(buys, sells, minimumTrade/totalFunds)
    
    return buys, sells, msrPort

def maxLogSigRatio(params):
    def Statistic(params):
        dataDict = params['dataDict']
        portfolio = params['portfolio']
        r = ff.portfolioReturn(dataDict, portfolio)
        v = ff.portfolioVolatility(dataDict, portfolio)
        return ff.myLog(r) - ff.myLog(v)
    
    #Dictionary of all the 
    dataDict = params['dataDict']
    #Time which the data will look back from the last entry and calculate the return and volatility (minutes)
    lookbackTime = params['lookBackTime']
    #Smallest amount in dollars a trade is allowed to happen in.
    minimumTrade = params['minimumTrade']
    #The current portfolio when this was called.
    currentPortfolio = params['currentPortfolio']
    #The total ammount of funds available (in USD)
    totalFunds = params['totalFunds']
    #Setting number of weights for the portfolio calculation
    params['numberOfWeights'] = len(dataDict)
    
    #List of assets is the indexes of dataDict
    for key in dataDict:
        dataDict[key] = dataDict[key][max(dataDict[key].index) - datetime.timedelta(minutes = lookbackTime):]
    
    #msrPort = ff.maximumSharpeRatio(dataDict.copy())
    msrPort = ff.maximizeStatistic(Statistic, params)
    
    msrPort = ff.nicefyPortfolio(msrPort, treshold = minimumTrade/totalFunds)
    
    porA = currentPortfolio.copy()
    porB = msrPort.copy()
    
    buys = {}
    sells = {}
    for key in porA.keys():
        if porB[key] > porA[key]:
            buys[key] = porB[key] - porA[key]
        elif porB[key] < porA[key]:
            sells[key] = porA[key] - porB[key]
            
    buys, sells = ff.nicefyTrades(buys, sells, minimumTrade/totalFunds)
    
    return buys, sells, msrPort