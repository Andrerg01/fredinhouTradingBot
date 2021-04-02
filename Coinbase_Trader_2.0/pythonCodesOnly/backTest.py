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
import argparse
import configparser

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

parser = argparse.ArgumentParser(description = 'Input Parameters')
parser.add_argument('-c', '--configFile', type = str, metavar = '',  help = 'Path to configuration file.')
parser.add_argument('-d', '--datesFile', type = str, metavar = '',  help = 'Path to test dates file.')
parser.add_argument('-p', '--paramsFile', type = str, metavar = '',  help = 'Path to parameters file.')

args = parser.parse_args()
config = configparser.ConfigParser()
try:
    config.read(args.configFile)
except:
    print("Failed to load config file.")
    exit()
try:
    testDates = pd.read_csv(args.datesFile, parse_dates=['Start', 'End'])
    testDates = testDates.drop('Unnamed: 0', axis = 1)
except:
    print("Failed to load dates file.")
    exit()
try:
    with open(args.paramsFile, 'rb') as f:
        params = pkl.load(f)
except:
    print("Failed to load parameters file.")
    exit()

testDates = [[testDates.iloc[i]['Start'],testDates.iloc[i]['End']] for i in range(len(testDates))]

strategy = config["Configurations"]["strategy"]
if strategy == 'maxSharpeRatio':
    Strategy = sg.maxSharpeRatio
elif strategy == 'maxLogSigRatio':
    Strategy = sg.maxLogSigRatio
elif strategy == 'equalWeights':
    Strategy = sg.equalWeights
else:
    print("Unknown startegy provided")
    exit()
    
header = config["Configurations"]["header"]
tradingFeePct = eval(config["Configurations"]["tradingFeePct"])
granularity = eval(config["Configurations"]["granularity"])
startingFunds = eval(config["Configurations"]["startingFunds"])
minimumTrade = eval(config["Configurations"]["minimumTrade"])
outputFile = config["Configurations"]["outputFile"]
allowedAssets = eval(config["Configurations"]["allowedAssets"])
cycleTime = eval(config["Configurations"]["cycleTime"])
extraEarlyTime = eval(config["Configurations"]["extraEarlyTime"])

fundsPerTime = {}
i = 0.
timeStart1 = datetime.datetime.now()

for date in testDates:
    epochLength = (date[1] - date[0]).total_seconds()
    timeStart2 = datetime.datetime.now()
    pct1 = i/len(testDates)*100.
    header1 = uf.progressBar(pct1, time0 = timeStart1, header = header + "\n") + "\n"
    fundsPerTime[str(date)] = []
    goodAssets = []
    for asset in allowedAssets:
        if ff.availableData(asset, date[0]) and ff.availableData(asset, date[1]):
            goodAssets += [asset]
    goodAssets = goodAssets + ['USD-USD']

    currentPortfolioSize = {asset:0. for asset in goodAssets}

    currentPortfolioSize['USD-USD'] = startingFunds

    start = date[0] - datetime.timedelta(seconds = extraEarlyTime)
    end = date[1]

    dataDict = cf.getData(Client, goodAssets, start = start, end = end, granularity = granularity, verbose = True, header = 'Gathering data.\n')

    dataDict = ff.nicefyData(dataDict, granularity = granularity, header = 'Nicefying data.\n', verbose = True)

    currentDate = date[0]
    while currentDate < date[1]:
        pct2 = ((currentDate - date[0]).total_seconds())/(date[1] - date[0]).total_seconds()*100.
        uf.clear()
        dataDictSoFar = {}
        for key in dataDict.keys():
            dataDictSoFar[key] = dataDict[key][:currentDate]
        totalFunds = 0
        for key in currentPortfolioSize.keys():
            totalFunds += currentPortfolioSize[key]*dataDictSoFar[key]['Close'][-1]
        print(str(totalFunds) + "\n" + str(currentDate))
        currentPortfolio = {}
        for key in currentPortfolioSize.keys():
            currentPortfolio[key] = currentPortfolioSize[key]*dataDictSoFar[key]['Close'][-1]/totalFunds

        params['dataDict'] = dataDictSoFar.copy()

        params['minimumTrade'] = minimumTrade
        #The current portfolio when this was called.
        params['currentPortfolio'] = currentPortfolio
        #The total ammount of funds available (in USD)
        params['totalFunds'] = totalFunds

        header2 = header1 + "Analyzing data from " + str(date[0]) + " to " + str(date[1]) + "\n"
        print(uf.progressBar(pct2, time0 = timeStart2, header = header2))
        print("Current Portfolio: " + ff.printPortfolio(currentPortfolio))

        buys, sells, bestPortfolio = Strategy(params)
        for key in buys.keys():
            buys[key] = buys[key]*totalFunds/(dataDictSoFar[key]['Close'][-1])
        for key in sells.keys():
            sells[key] = sells[key]*totalFunds/(dataDictSoFar[key]['Close'][-1])
        currentPortfolioSize = makeFakeTrades(buys, sells, currentPortfolioSize, dataDictSoFar, tradingFeePct)        
        currentDate += datetime.timedelta(seconds = cycleTime)
        fundsPerTime[str(date)] += [totalFunds]
    i += 1

with open(outputFile, 'wb') as f:
    pkl.dump(fundsPerTime, f)