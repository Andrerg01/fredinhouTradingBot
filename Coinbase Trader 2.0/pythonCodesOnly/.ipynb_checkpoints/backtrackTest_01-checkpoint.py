import cbpro
import pandas as pd
import pickle as pkl
import datetime
import matplotlib.pyplot as plt
import financialFunctions as ff
import coinbaseFunctions as cf
import utilityFunctions as uf
import numpy as np
import matplotlib.patches as patches
from decimal import *
import os
from os import listdir
from os.path import isfile, join
import Strategies as sg

def makeFakeTrades(buys, sells, portfolioSize, dataDict, tradingFeePct):
    for key in sells.keys():
        if key != 'USD-USD':
            portfolioSize[key] -= sells[key]
            portfolioSize['USD-USD'] += sells[key]*dataDict[key]['Close'][-1]*(1-tradingFeePct)
            
    for key in buys.keys():
        if key != 'USD-USD':
            portfolioSize[key] += buys[key]
            portfolioSize['USD-USD'] -= buys[key]*dataDict[key]['Close'][-1]*(1-tradingFeePct)
            
    return portfolioSize

with open('../../../fredinhouTradingBot_Pvt/coinbase_credentials_1.pkl', 'rb') as f:
    credentials = pkl.load(f)
Client = cbpro.AuthenticatedClient(
    credentials['APIKey'],
    credentials['SecretKey'],
    credentials['passPhrase'],
    api_url = credentials['APIurl']
    )
testingDates = [[datetime.datetime.now() - datetime.timedelta(days = 3*356 + int(r)), datetime.datetime.now() - datetime.timedelta(days = 3*356 + int(r) - 60)] for r in np.random.random(size = 10)*120]

#Backtracking
downloadedAssets = ['AAVE-USD', 'ADA-USD', 'ALGO-USD', 'ATOM-USD', 'BAL-USD', 'BAND-USD', 'BCH-USD', 'BNT-USD', 'BTC-USD', 'CGLD-USD', 'COMP-USD', 'DASH-USD', 'EOS-USD', 'ETC-USD', 'ETH-USD', 'FIL-USD', 'GRT-USD', 'KNC-USD', 'LINK-USD', 'LRC-USD', 'LTC-USD', 'MATIC-USD', 'MKR-USD', 'NMR-USD', 'NU-USD', 'OMG-USD', 'OXT-USD', 'REN-USD', 'REP-USD', 'SKL-USD', 'SNX-USD', 'SUSHI-USD', 'UMA-USD', 'UNI-USD','WBTC-USD', 'XLM-USD', 'XTZ-USD', 'YFI-USD', 'ZEC-USD', 'ZRX-USD']
now = datetime.datetime.now()
testingTimeInterval = [now - datetime.timedelta(days = 3*365), now - datetime.timedelta(days = 1*365)]
epochDuration = 30*24*60*60
numberOfEpochs = 10
maxTimeDelta = (testingTimeInterval[-1]-testingTimeInterval[0]).total_seconds() - 3*epochDuration
testTimes = []
intervalBetweenDataGather = 60*60
for i in range(numberOfEpochs):
    t1 = testingTimeInterval[0] + datetime.timedelta(seconds = np.random.randint(low = epochDuration,high = maxTimeDelta))
    t2 = t1 + datetime.timedelta(seconds = int(2.1*epochDuration))
    testTimes += [[t1,t2]]

fundsPerTime = {}
i = 0.
timeStart1 = datetime.datetime.now()
for times in testTimes:
    timeStart2 = datetime.datetime.now()
    pct1 = i/len(testTimes)*100.
    header1 = uf.progressBar(pct1, time0 = timeStart1) + "\n"
    fundsPerTime[str(times)] = []
    goodAssets = []
    for asset in downloadedAssets:
        if ff.availableData(asset, times[0]) and ff.availableData(asset, times[1]):
            goodAssets += [asset]
    goodAssets = goodAssets + ['USD-USD']
    tradingFeePct = 0.005
    currentPortfolioSize = {asset:0. for asset in goodAssets}
    currentPortfolioSize['USD-USD'] = 100.
    dataDict = cf.getData(Client, goodAssets, start = times[0], end = times[1], granularity = 60, verbose = False, header = '')
    currentDate = times[0] + datetime.timedelta(seconds = epochDuration)
    iniDiff = (currentDate - times[0]).total_seconds()
    while (currentDate - times[0]).total_seconds() < 2*epochDuration:
        pct2 = ((currentDate - times[0]).total_seconds()-iniDiff)/(2*epochDuration)*100.
        uf.clear()
        print(uf.progressBar(pct2, time0 = timeStart2, header = header1))
        dataDictSoFar = {}
        for key in dataDict.keys():
            dataDictSoFar[key] = dataDict[key][:currentDate]
        totalFunds = 0
        for key in currentPortfolioSize.keys():
            totalFunds += currentPortfolioSize[key]*dataDictSoFar[key]['Close'][-1]
        print(totalFunds)
        currentPortfolio = {}
        for key in currentPortfolioSize.keys():
            currentPortfolio[key] = currentPortfolioSize[key]*dataDictSoFar[key]['Close'][-1]/totalFunds

        params = {}
        params['dataDict'] = dataDictSoFar.copy()
        params['lookBackTime'] = 60*60*24
        params['minimumTrade'] = 5
        #The current portfolio when this was called.
        params['currentPortfolio'] = currentPortfolio
        #The total ammount of funds available (in USD)
        params['totalFunds'] = totalFunds
        buys, sells, bestPortfolio = sg.maxSharpeRatio(params)
        for key in buys.keys():
            buys[key] = buys[key]*totalFunds/(dataDictSoFar[key]['Close'][-1])
        for key in sells.keys():
            sells[key] = sells[key]*totalFunds/(dataDictSoFar[key]['Close'][-1])
        currentPortfolioSize = makeFakeTrades(buys, sells, currentPortfolioSize, dataDictSoFar, tradingFeePct)        
        currentDate += datetime.timedelta(seconds = intervalBetweenDataGather)
        fundsPerTime[str(times)] += [totalFunds]

    i += 1

with open("backtrackRestult_01.pkl", 'wb') as f:
    pkl.dump(fundsPerTime, f)