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
import financialFunctions as ff
import coinbaseFunctions as cf
import utilityFunctions as uf

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
#assets = ['ALGO-USD', 'ATOM-USD', 'BCH-USD', 'BTC-USD', 'DASH-USD', 'EOS-USD', 'ETC-USD', 'ETH-USD', 'KNC-USD', 'LINK-USD', 'LTC-USD', 'OXT-USD', 'REP-USD', 'XLM-USD', 'XTZ-USD']
#Ordered by score
#assets = ['REP-USD', 'OXT-USD', 'EOS-USD', 'XTZ-USD', 'KNC-USD', 'ALGO-USD', 'ATOM-USD', 'DASH-USD', 'XLM-USD', 'ETC-USD', 'BCH-USD', 'LINK-USD', 'LTC-USD', 'BTC-USD', 'ETH-USD']
assets = ['ETH-USD', 'BTC-USD', 'LTC-USD', 'LINK-USD', 'BCH-USD', 'ETC-USD', 'XLM-USD', 'DASH-USD', 'ATOM-USD', 'ALGO-USD', 'KNC-USD', 'XTZ-USD', 'EOS-USD', 'OXT-USD', 'REP-USD']

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
runTrades = False

try:
    with open("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/portfolio21600.pkl",'rb') as f:
        portfolio21600 = pkl.load(f)
except:
    portfolio21600 = {asset:0 for asset in (assets + ['USD-USD'])}
    
try:
    with open("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/checkpointSMS.pkl",'rb') as f:
        checkpointSMS = pkl.load(f)
except:
    checkpointSMS = 200
    
timeRun = datetime.datetime.now()
timeTrades = datetime.datetime.now()

buyLimit = 4

while True:
    header = \
"""
######################################################
#                  Auto Trader 4.0                   #
#       Author: """ + "Andre Guimaraes" + " "*(37 - len("Andre Guimaraes")) + """#
######################################################
"""
    end = datetime.datetime.now()
    start = end - datetime.timedelta(seconds = 60*60*24*365)

    if run:
        header += "Loading data for all assets at granularity " + str(60*60) + ".\n\n"
        os.system("clear")
        print(header)
    
        fail = True
        while(fail):
            try:
                allData3600 = [cf.getData(Client, asset, start, end, granularity = 60*60) for asset in assets]
                fail = False
            except:
                pass
        allCloseData3600 = np.array([d['close'].values for d in allData3600])

        timeRun = datetime.datetime.now() 
        
        header += "Downloading current Client Information.\n\n"
        os.system("clear")
        print(header)
        
        currentPrices = {assets[i]:allCloseData3600[i][-1] for i in range(len(assets))}
        currentPrices = [currentPrices[key] for key in assets] + [1]
        
        fundsSize = cf.getFunds(Client, assets, allCloseData3600, size = True)
        fundsSize = [fundsSize[key] for key in assets] + [fundsSize['USD-USD']]
        funds = [fundsSize[i]*currentPrices[i] for i in range(len(assets)+1)]
        totalFunds = sum(funds)
        
        negotiableFunds = 0.9*totalFunds

        buys = {}
        sells = {}
        
        if runTrades:
            header += "Preparing to trade\n\n"
            os.system("clear")
            print(header)
            
            granularity = 6*60*60
            timeTrades = datetime.datetime.now()

            header += "Loading data fot all assets at granularity " + str(granularity) + ".\n\n"
            os.system("clear")
            print(header)
            
            fail = True
            while(fail):
                try:
                    allData21600 = [cf.getData(Client, asset, start, end, granularity = granularity) for asset in assets]
                    fail = False
                except:
                    pass
            allCloseData21600 = np.array([d['close'].values for d in allData21600])
            currentScores = []

            header += "Calculating strategy curves.\n\n"
            os.system("clear")
            print(header)
            
            buyHistB = []
            for i in range(len(assets)):
                with open(cf.dbPath + "/" + assets[i] + "/Str" + 'B' + "Params_" + str(granularity) + ".pkl", 'rb') as f:
                    best = pkl.load(f)
                    parameters = best['parameters']
                    score = best['score']
                nLims = np.array([2,max([len(data) for data in allCloseData21600])])
                buyHistB += [ff.backtestStrategyB(allCloseData21600[i], parameters[0], parameters[1], parameters[2], nLims)[2]]

            buyHistC = []
            for i in range(len(assets)):
                with open(cf.dbPath + "/" + assets[i] + "/Str" + 'C' + "Params_" + str(granularity) + ".pkl", 'rb') as f:
                    best = pkl.load(f)
                    parameters = best['parameters']
                    score = best['score']
                nLims = np.array([2,max([len(data) for data in allCloseData21600])])
                buyHistC += [ff.backtestStrategyC(allCloseData21600[i], parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], nLims)[2]]

            buyHist = [[0 for j in range(len(buyHistB[i]))] for i in range(len(buyHistB))]
            for i in range(len(buyHistB)):
                for j in range(len(buyHistB[i])):
                    buyHist[i][j] = buyHistB[i][j] and buyHistC[i][j]
          
            buyState = [funds[i] > 0.5*1/buyLimit*negotiableFunds for i in range(len(assets))]
            header += "Buy States: " + str([assets[i] + ": " + str(buyState[i]) for i in range(len(assets))]) + "\n"
            
            header += "Checking for buy/sell signals.\n\n"
            os.system("clear")
            print(header)
            
            #Sells
            for j in range(len(assets)):
                if not buyHist[j][-2] and buyState[j]:
                    header += "Selling "  + assets[j] + ".\n"
                    header += str(cf.sell(Client, assets[j], fundsSize[j])) + "\n\n"
                    sells[assets[i]] = fundsSize[j]
                    os.system("clear")
                    print(header)
            
            #Buys
            for j in range(len(assets)):
                buyState = [funds[i] > 0.1*1/buyLimit*negotiableFunds for i in range(len(assets))]
                buyFull = sum(buyState) >= buyLimit
                if not buyHist[j][-3] and buyHist[j][-2] and not buyState[j] and not buyFull:
                    header += "Purchasing " + assets[j] + ".\n"
                    header += str(cf.buy(Client, assets[j], negotiableFunds*1/buyLimit/currentPrices[j])) + "\n\n"
                    buys[assets[i]] = negotiableFunds*1/buyLimit/currentPrices[j]
                    os.system("clear")
                    print(header)
                
                    
            header += "Done with trades for now.\n\n"
            os.system("clear")
            print(header)
            runTrades = False
               
        
        header += "Making hourly calculations.\n\n"
        os.system("clear")
        print(header)
        
        currentPortfolio = cf.getPortfolio(Client, assets, allCloseData3600)
        
        header += "Current total funds: $" + str(totalFunds) + "\n\n"
        header += "Current portfolio: " + ff.printPortfolio(currentPortfolio) + "\n\n"
        os.system("clear")
        print(header)
        
        currentPrices = {assets[i]:allCloseData3600[i][-1] for i in range(len(assets))}
        buys = {}
        sells = {}
        
        header += "Saving information and making plots.\n\n"
        os.system("clear")
        print(header)
        
        hist = pd.read_csv("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/hist.csv", parse_dates=['Date'], converters = {'Portfolio': eval}).set_index('Date')
        
        currentPortfolio = cf.getPortfolio(Client, assets, allCloseData3600)
        
        hist = hist.append(pd.DataFrame({'Date':[datetime.datetime.now()], 'Funds':[totalFunds], 'Portfolio':[currentPortfolio], 'Rebalance':[False], 'Buys':[buys], 'Sells':[sells], 'Deposits':[hist.iloc[-1]['Deposits']]}).set_index('Date'))
        
        hist.to_csv("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/hist.csv")
        histTemp = hist[datetime.datetime.now() - datetime.timedelta(days = 30):].copy()
        fig, ax = plt.subplots(2, 2, figsize = [20,20/1.61])
        try:
            ff.makePricePlot(histTemp, ax[0, 0])
        except:
            pass
        try:
            ff.makeTimePortfolioPlot(histTemp, ax[0, 1])
        except:
            pass
        try:
            ff.makeMarketPerformancePlot([d[histTemp.index[0]:]['close'].values for d in allData3600], ax[1, 0])
        except:
            pass
        try:
            ff.makeCurrentPortfolioPlot(histTemp, ax[1, 1])
        except:
            pass
        
        fig.tight_layout()
        
        fig.savefig("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/PortfolioPlot.png")
        #fig.savefig("/var/www/html/PortfolioPlot.png")
        sh.copy("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/PortfolioPlot.png", "/var/www/html/assets/images/portfolioplot-1646x1022.png")
        
        
        if totalFunds - checkpointSMS >= totalFunds*0.05:
            clientTwilio.messages.create(to="+19363332711", from_="+16362491689", body="OH YEAH! You just made $" + str(abs(totalFunds - checkpointSMS)) + "!\n Let's be rich! (:\nCurrent Balance: " + str(totalFunds))
            checkpointSMS = totalFunds
            with open("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/checkpointSMS.pkl",'wb') as f:
                 pkl.dump(checkpointSMS, f)
        elif totalFunds - checkpointSMS <= -totalFunds*0.05:
            clientTwilio.messages.create(to="+19363332711", from_="+16362491689", body="OH NO! You just lost $" + str(abs(totalFunds - checkpointSMS)) + "! We're gonna be poor! ):\nCurrent Balance: " + str(totalFunds))
            checkpointSMS = totalFunds
            with open("/home/andrerg01/AutoTraders/fredinhouTradingBot/Coinbase_Trader_4.0/logs/checkpointSMS.pkl",'wb') as f:
                 pkl.dump(checkpointSMS, f)
        
        header += "-----Sales done! Waiting for next trade window.-----"
        os.system("clear")
        print(header) 
        run = False
    
    time.sleep(30)
    #To repeat every 15 minutes
    if datetime.datetime.now().minute % 5 == 0 and datetime.datetime.now().minute != timeTrades.minute:
        run = True
    #To repeat every 6 hours
    if (datetime.datetime.now().hour == 0 or datetime.datetime.now().hour == 6 or datetime.datetime.now().hour == 12 or datetime.datetime.now().hour == 18) and datetime.datetime.now().hour != timeTrades.hour:
        runTrades = True
    

    
    
