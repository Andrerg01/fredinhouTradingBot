import cbpro
import pandas as pd
import pickle as pkl
import datetime
import matplotlib.pyplot as plt
import financialFunctions as ff
import coinbaseFunctions as cf
import utilityFunctions as uf
import Strategies as sg
import Strategies as sg
import numpy as np
import time
import matplotlib.patches as patches
import math
from decimal import *
import argparse
import configparser

assets = ['AAVE-USD', 'ADA-USD', 'ALGO-USD', 'ATOM-USD', 'BAL-USD', 'BAND-USD', 'BCH-USD', 'BNT-USD', 'BTC-USD', 'CGLD-USD', 'COMP-USD', 'DASH-USD', 'EOS-USD', 'ETC-USD', 'ETH-USD', 'FIL-USD', 'GRT-USD', 'KNC-USD', 'LINK-USD', 'LRC-USD', 'LTC-USD', 'MATIC-USD', 'MKR-USD', 'NMR-USD', 'NU-USD', 'OMG-USD', 'OXT-USD', 'REN-USD', 'REP-USD', 'SKL-USD', 'SNX-USD', 'SUSHI-USD', 'UMA-USD', 'UNI-USD', 'USD-USD', 'WBTC-USD', 'XLM-USD', 'XTZ-USD', 'YFI-USD', 'ZEC-USD', 'ZRX-USD']

loopTime = datetime.timedelta(hours = 1)

with open('../../../fredinhouTradingBot_Pvt/coinbase_credentials.pkl', 'rb') as f:
    credentials = pkl.load(f)
Client = cbpro.AuthenticatedClient(
    credentials['APIKey'],
    credentials['SecretKey'],
    credentials['passPhrase'],
    api_url = credentials['APIurl']
    )

parser = argparse.ArgumentParser(description = 'Input Parameters')
parser.add_argument('-c', '--configFile', type = str, metavar = '', required = True, help = 'Path to configuration file.')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.configFile)

strategy = config["Configurations"]["strategy"]    
header = config["Configurations"]["header"]
tradingFeePct = eval(config["Configurations"]["tradingFeePct"])
granularity = eval(config["Configurations"]["granularity"])
minimumTrade = eval(config["Configurations"]["minimumTrade"])
outputPath = config["Configurations"]["outputPath"]
cycleTime = config["Configurations"]["cycleTime"]
extraEarlyTime = eval(config["Configurations"]["extraEarlyTime"])
allowedAssets = eval(config["Configurations"]["allowedAssets"])
params = eval(config["Configurations"]["params"])

if strategy == 'maxSharpeRatio':
    Strategy = sg.maxSharpeRatio
elif strategy == 'maxLogSigRatio':
    Strategy = sg.maxLogSigRatio
elif strategy == 'equalWeights':
    Strategy = sg.equalWeights
else:
    print("Unknown startegy provided, check configuration file.")
    exit()
    
author = 'Andre Guimaraes'
header = \
"""
######################################################
#                  Auto Trader 2.0                   #
#       Author: """ + author + " "*(37 - len(author)) + """#
#       Strategy: """ + strategy + " "*(35 - len(strategy)) + """#
######################################################
"""
while(True):
    time0 = datetime.datetime.now()
    
    header += '\nUpdating Data...\n'
    #Unskip for real application, commented for test
    #cf.updateData(Client, assets, granularity = granularity, verbose = True, header = header)
    header += 'Done\n'
    
    end = datetime.datetime.now()
    start = end - datetime.timedelta(seconds = (params['lookBackTime'] + extraEarlyTime))
    header += "Gathering current data.\n"
    uf.clear()
    print(header)
    dataDict = cf.getData(Client, assets, start = start, end = end, granularity = granularity, verbose = True, header = header)
    header += "Done\n"
    header += "Filling holes in data\n"
    uf.clear()
    print(header)

    lengths = []
    keys = []
    for key in dataDict.keys():
        lengths += [len(dataDict[key])]
        keys += [key]
    meanL = np.mean(lengths)
    stdL = np.std(lengths)
    for key in keys:
        #Only data with length withing 1 std may pass!
        if len(dataDict[key]) < meanL - stdL:
            del dataDict[key]
    dataDict = ff.nicefyData(dataDict, granularity, header = header)
    
    header += "Done\n"
    header += "Calculating optimal portfolio and corresponding buys/sells.\n"
    uf.clear()
    print(header)
    portfolioUSD = cf.getFunds(Client, dataDict, coin = 'funds')
    portfolioSIZE = cf.getFunds(Client, dataDict, coin = 'size')
    currentPortfolio = cf.getPortfolio(Client, dataDict)
    header += "Done\n"

    totalFunds = sum([portfolioUSD[key] for key in portfolioUSD.keys()])
    header += "Current Total Funds(USD): " + str(totalFunds) + "\n"
    uf.clear()
    print(header)
    
    params['dataDict'] = dataDict.copy()
    params['minimumTrade'] = minimumTrade
    params['currentPortfolio'] = currentPortfolio
    params['totalFunds'] = totalFunds
    
    buys, sells, targetPortfolio = Strategy(params)

    print(buys)
    print(sells)
    print(targetPortfolio)
    
    exit()
    
    
    
    

    
    
    
    
    timeEllapsed = (datetime.datetime.now() - time0).total_seconds()
    
    time.sleep(cycleTime - timeEllapsed)

#Making code to update data automatically
#dataDict = cf.getData(Client, )



