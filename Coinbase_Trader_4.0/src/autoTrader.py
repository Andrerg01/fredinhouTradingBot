print("Initializing Packages")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cbpro
import pandas as pd
import pickle as pkl
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from numba import njit
from concurrent.futures import ThreadPoolExecutor
import gc
gc.enable()
import sys
import shutil as sh
from twilio.rest import Client as twClient

#My Libraries
import coinbaseFunctions as cf
import utilityFunctions as uf
import financialFunctions as ff

def calculateStrategyPortfolio(allCloseData, assets, strategyLetter, granularity):
    parameters = []
    scores = []
    for asset in assets:
        with open(cf.dbPath + "/" + asset + "/Str" + strategyLetter + "Params_" + str(granularity) + ".pkl", 'rb') as f:
            best = pkl.load(f)
            parameters += [best['parameters']]
            scores += [best['score']]
    parameters = np.array(parameters)
    nLims = np.array([2,max([len(data) for data in allCloseData])])
    if strategyLetter == 'A':
        currentScore = np.array([ff.backtestStrategyA(allCloseData[i], parameters[i][0], parameters[i][1], parameters[i][2], nLims)[2][-1]*scores[i] for i in range(len(allCloseData))])
    elif strategyLetter == 'B':
        currentScore = np.array([ff.backtestStrategyB(allCloseData[i], parameters[i][0], parameters[i][1], parameters[i][2], nLims)[2][-1]*scores[i] for i in range(len(allCloseData))])
    elif strategyLetter == 'C':
        currentScore = np.array([ff.backtestStrategyC(allCloseData[i], parameters[i][0], parameters[i][1], parameters[i][2], parameters[i][3], parameters[i][4], nLims)[2][-1]*scores[i] for i in range(len(allCloseData))])
    return currentScore

#Assets to be considered for purchasing
assets = ['AAVE-USD', 'ADA-USD', 'ALGO-USD', 'ATOM-USD', 'BAL-USD', 'BAND-USD', 'BCH-USD', 'BNT-USD', 'BTC-USD', 'CGLD-USD', 'COMP-USD',\
                 'DASH-USD', 'EOS-USD', 'ETC-USD', 'ETH-USD', 'FIL-USD', 'GRT-USD', 'KNC-USD', 'LINK-USD', 'LRC-USD', 'LTC-USD', 'MATIC-USD',\
                 'MKR-USD', 'NMR-USD', 'NU-USD', 'OMG-USD', 'OXT-USD', 'REN-USD', 'REP-USD', 'SKL-USD', 'SNX-USD', 'SUSHI-USD', 'UMA-USD',
                 'UNI-USD','WBTC-USD', 'XLM-USD', 'XTZ-USD', 'YFI-USD', 'ZEC-USD', 'ZRX-USD']

with open('/home/andrerg01/AutoTraders/fredinhouTradingBot_Pvt/coinbase_credentials.pkl', 'rb') as f:
    credentials = pkl.load(f)
Client = cbpro.AuthenticatedClient(
    credentials['APIKey'],
    credentials['SecretKey'],
    credentials['passPhrase'],
    api_url = credentials['APIurl']
    )

clientTwilio = twClient("AC4ed06be97b80927880222036093c6320", "a8410533928c1b06520cecec7ae3f6c9")


run = False
run1 = False
run2 = False
run3 = False
try:
    with open("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/portfolio3600.pkl",'rb') as f:
        portfolio3600 = pkl.load(f)
    with open("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/portfolio21600.pkl",'rb') as f:
        portfolio21600 = pkl.load(f)
    with open("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/portfolio86400.pkl",'rb') as f:
        portfolio86400 = pkl.load(f)
except:
    portfolio86400 = {asset:0 for asset in (assets + ['USD-USD'])}
    portfolio21600 = {asset:0 for asset in (assets + ['USD-USD'])}
    portfolio3600 = {asset:0 for asset in (assets + ['USD-USD'])}
    
try:
    with open("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/checkpointSMS.pkl",'rb') as f:
        checkpointSMS = pkl.load(f)
except:
    checkpointSMS = 100
time1 = datetime.datetime.now()
time2 = datetime.datetime.now()
time3 = datetime.datetime.now()
while True:
    header = \
"""
######################################################
#                  Auto Trader 4.0                   #
#       Author: """ + "Andre Guimaraes" + " "*(37 - len("Andre Guimaraes")) + """#
######################################################
"""
    start = datetime.datetime(2016,3,27)
    end = datetime.datetime.now()

    if run1:
        granularity = 60*60

        header += "Loading data for all assets at granularity " + str(granularity) + ".\n\n"
        os.system("clear")
        print(header)

        allData3600 = [cf.getData(Client, asset, start, end, granularity = granularity) for asset in assets]
        allCloseData3600 = np.array([d['close'].values for d in allData3600])

        currentScores = []
        for strategyLetter in ['A', 'B', 'C']:
            header += "Loading optimal parameters for strategy " + strategyLetter + " at granularity " + str(granularity) + ".\n\n"
            os.system("clear")
            print(header)
            currentScores += [calculateStrategyPortfolio(allCloseData3600, assets, strategyLetter, granularity)]
        totalScores = sum(currentScores)
        #USD score is at least as great as the maximum score
        totalScores = np.append(totalScores, [np.max(totalScores)])
        portfolio3600 = {(assets + ['USD-USD'])[i]:totalScores[i] for i in range(len(assets + ['USD-USD']))}
        
        run1 = False
        time1 = datetime.datetime.now()
        
        with open("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/portfolio3600.pkl",'wb') as f:
            pkl.dump(portfolio3600, f)
        
        
    if run2:
        
        granularity = 6*60*60

        header += "Loading data for all assets at granularity " + str(granularity) + ".\n\n"
        os.system("clear")
        print(header)

        allData21600 = [cf.getData(Client, asset, start, end, granularity = granularity) for asset in assets]
        allCloseData21600 = np.array([d['close'].values for d in allData21600])

        currentScores = []
        for strategyLetter in ['A', 'B', 'C']:
            header += "Loading optimal parameters for strategy " + strategyLetter + " at granularity " + str(granularity) + ".\n\n"
            os.system("clear")
            print(header)
            currentScores += [calculateStrategyPortfolio(allCloseData21600, assets, strategyLetter, granularity)]
        totalScores = sum(currentScores)
        #USD score is at least as great as the maximum score
        totalScores = np.append(totalScores, [np.max(totalScores)])
        portfolio21600 = {(assets + ['USD-USD'])[i]:totalScores[i] for i in range(len(assets + ['USD-USD']))}
        
        run2 = False
        time2 = datetime.datetime.now()
        
        with open("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/portfolio21600.pkl",'wb') as f:
            pkl.dump(portfolio21600, f)
        
    if run3:
        
        granularity = 24*60*60

        header += "Loading data for all assets at granularity " + str(granularity) + ".\n\n"
        os.system("clear")
        print(header)

        allData86400 = [cf.getData(Client, asset, start, end, granularity = granularity) for asset in assets]
        allCloseData86400 = np.array([d['close'].values for d in allData86400])

        currentScores = []
        for strategyLetter in ['A', 'B', 'C']:
            header += "Loading optimal parameters for strategy " + strategyLetter + " at granularity " + str(granularity) + ".\n\n"
            os.system("clear")
            print(header)
            currentScores += [calculateStrategyPortfolio(allCloseData86400, assets, strategyLetter, granularity)]
        totalScores = sum(currentScores)
        #USD score is at least as great as the maximum score
        totalScores = np.append(totalScores, [np.max(totalScores)])
        portfolio86400 = {(assets + ['USD-USD'])[i]:totalScores[i] for i in range(len(assets + ['USD-USD']))}   
        
        run3 = False
        time3 = datetime.datetime.now()
        
        with open("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/portfolio86400.pkl",'wb') as f:
            pkl.dump(portfolio86400, f)
            
    if run:
        finalPortfolio = {asset:0 for asset in (assets+['USD-USD'])}
        for key in finalPortfolio.keys():
            finalPortfolio[key] = portfolio3600[key] + portfolio21600[key] + portfolio86400[key]
        finalPortfolio = ff.nicefyPortfolio(finalPortfolio, 0.03)

        header += "Final Portfolio: " + ff.printPortfolio(finalPortfolio) + ".\n\n"
        os.system("clear")
        print(header)
        
        header += "Loading current funds and portfolio.\n\n"
        os.system("clear")
        print(header)
        
        funds = cf.getFunds(Client, assets, allCloseData3600)
        totalFunds = sum([funds[key] for key in funds.keys()])
        currentPortfolio = cf.getPortfolio(Client, assets, allCloseData3600)
        
        header += "Current total funds: $" + str(totalFunds) + "\n\n"
        header += "Current portfolio: " + ff.printPortfolio(currentPortfolio) + "\n\n"
        os.system("clear")
        print(header)
        
        currentPrices = {assets[i]:allCloseData3600[i][-1] for i in range(len(assets))}
        
        buys = {}
        sells = {}
        for key in finalPortfolio.keys():
            if finalPortfolio[key] > currentPortfolio[key]:
                buys[key] = finalPortfolio[key] - currentPortfolio[key]
            elif finalPortfolio[key] < currentPortfolio[key]:
                sells[key] = currentPortfolio[key] - finalPortfolio[key]
        buys, sells = ff.nicefyTrades(buys, sells, 0.00)
        for key in buys.keys():
            if key != "USD-USD":
                buys[key] = buys[key]*totalFunds/currentPrices[key]
        for key in sells.keys():
            if key != "USD-USD":
                sells[key] = sells[key]*totalFunds/currentPrices[key]
        
        header += "Pruchases to be made: " + ff.printPortfolio(buys) + "\n\n"
        header += "Sales to be made: " + ff.printPortfolio(sells) + "\n\n"
        os.system("clear")
        print(header)
        
        cf.makeTrades(Client, buys, sells)
        
        header += "Saving information and making plots.\n\n"
        os.system("clear")
        print(header)
        
        hist = pd.read_csv("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/hist.csv", parse_dates=['Date'], converters = {'Portfolio': eval}).set_index('Date')
        
        currentPortfolio = cf.getPortfolio(Client, assets, allCloseData3600)
        
        hist = hist.append(pd.DataFrame({'Date':[datetime.datetime.now()], 'Funds':[totalFunds], 'Portfolio':[currentPortfolio], 'Rebalance':[False], 'Buys':[buys], 'Sells':[sells]}).set_index('Date'))
        
        hist.to_csv("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/hist.csv")
        
        fig, ax = plt.subplots(2, 2, figsize = [20,20/1.61])
        ff.makePricePlot(hist, ax[0, 0])
        ff.makeTimePortfolioPlot(hist, ax[0, 1])
        ff.makeCurrentPortfolioPlot(hist, ax[1, 0])
        ff.makeMarketPerformancePlot([d[hist.index[0]:hist.index[-1]]['close'].values for d in allData3600], ax[1, 1])
        fig.tight_layout()
        
        fig.savefig("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/PortfolioPlot.png")
        #fig.savefig("/var/www/html/PortfolioPlot.png")
        sh.copy("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/PortfolioPlot.png", "/var/www/html/PortfolioPlot.png")
        
        
        if totalFunds - checkpointSMS >= 10:
            clientTwilio.messages.create(to="+19363332711", from_="+16362491689", body="Woohoo! You just made $" + str(totalFunds - checkpointSMS) + "!\n Let's be rich! (:\nCurrent Balance: " + str(totalFunds))
            checkpointSMS = totalFunds
            with open("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/checkpointSMS.pkl",'wb') as f:
                 pkl.dump(checkpointSMS, f)
        elif totalFunds - checkpointSMS <= -10:
            clientTwilio.messages.create(to="+19363332711", from_="+16362491689", body="On no! You just lost $" + str(totalFunds - checkpointSMS) + "! We're gonna be poor! ):\nCurrent Balance: " + str(totalFunds))
            checkpointSMS = totalFunds
            with open("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/checkpointSMS.pkl",'wb') as f:
                 pkl.dump(checkpointSMS, f)
        
        header += "-----Sales done! Waiting for next hour.-----"
        os.system("clear")
        print(header) 
        run = False
    
    time.sleep(60)
    #To repeat every hour
    if abs(datetime.datetime.now().hour - time1.hour) >= 1:
        run1 = True
    if (datetime.datetime.now().hour == 0 or datetime.datetime.now().hour == 6 or datetime.datetime.now().hour == 12 or datetime.datetime.now().hour == 18) and datetime.datetime.now().hour != time2.hour:
        run2 = True
    if abs(datetime.datetime.now().day - time3.day) >= 1:
        run3 = True
    if run1 or run2 or run3:
        run = True
    

    
    
