import time
import cbpro
import numpy as np
import pandas as pd
import datetime
import utilityFunctions as uf
import financialFunctions as ff
import os
import pickle as pkl
from os import listdir
from os.path import isfile, join

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
        if not isinstance(candles, dict):
            success = True
        else:
            if verbose:
                print("Failure in retreiving data for " + asset + ". Waiting 1 second and trying again. " + str(tries + 1) + "/" + str(triesMax) + "\n" + str(candles))
            tries += 1
            time.sleep(.1)
    if not success:
        print("Error getting candles for " + asset)
        print(candles)
        return []
    else:
        return candles
    
def dataFromFile(asset, start = None, end = None, granularity = None, verbose = True):
    startStamp = int(datetime.datetime.timestamp(start))
    endStamp = int(datetime.datetime.timestamp(end))
    path = '../candlesDataBase/' + asset
    fileStarts = []
    fileEnds = []
    for file in listdir(path):
        if file[-3:] == 'csv' and eval(file.split('_')[3].split('.')[0]) == granularity:
            fileStarts += [eval(file.split('_')[1])]
            fileEnds += [eval(file.split('_')[2])]
    fileStarts.sort()
    fileEnds.sort()
    if len(fileStarts) == 0:
        return pd.DataFrame([[float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan')]], columns = ['Date', 'Low', 'High', 'Open', 'Close', 'Volume', 'Return']).set_index('Date').sort_index().dropna()
    goodFiles = []
    if startStamp < min(fileStarts):
        if endStamp <= min(fileStarts):
            if verbose: print("File too far back, not stored in data.")
            pass;
        else:
            if verbose: print("Some of the data is too far back, retrieving only stored portions.")
            for i in range(len(fileStarts)):
                if endStamp > fileStarts[i]:
                    goodFiles += [asset + '_' + str(fileStarts[i]) + '_' + str(fileEnds[i]) + "_" + str(granularity) + ".csv"]
    else:
        if verbose: print('All seems to be fine, gathering data from files.')
        iniFound = False
        for i in range(len(fileStarts)):
            if fileStarts[i] <= startStamp <= fileEnds[i]:
                goodFiles += [asset + '_' + str(fileStarts[i]) + '_' + str(fileEnds[i]) + "_" + str(granularity) + ".csv"]
                iniFound = True
            elif iniFound:
                goodFiles += [asset + '_' + str(fileStarts[i]) + '_' + str(fileEnds[i]) + "_" + str(granularity) + ".csv"]
                
            if fileStarts[i] <= endStamp <= fileEnds[i]:
                iniFound = False
    data = pd.DataFrame([[float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan')]], columns = ['Date', 'Low', 'High', 'Open', 'Close', 'Volume', 'Return']).set_index('Date').sort_index().dropna()
    for file in goodFiles:
        #with open(path + '/' + file, 'rb') as f:
        #    dataToAppend = pkl.load(f)
        dataToAppend = pd.read_csv(path + '/' + file, parse_dates=['Date'])
        #dataToAppend.reset_index(inplace = True)
        if str(list(dataToAppend.columns)) == str(['time','low','high','open','close','volume']):
            dataToAppend.columns = ['Date','Low','High','Open','Close','Volume']
            dataToAppend = dataToAppend.set_index('Date')
        if 'Return' in list(dataToAppend.columns):
            dataToAppend = dataToAppend.drop("Return", axis = 1)
        for column in list(dataToAppend.columns):
            if 'Unnamed' in column or 'index' in column or 'level' in column:
                dataToAppend = dataToAppend.drop(column, axis = 1)
        dataToAppend = ff.concatReturns(dataToAppend).sort_index()
        data.reset_index(inplace = True)
        #dataToAppend.reset_index(inplace = True)
        data = data.append(dataToAppend).dropna()
        data = data.set_index('Date').sort_index()
    if 'Return' not in list(data.columns):
        data = ff.concatReturns(data).sort_index()
    return data[start:end].sort_index()
            
def getData(Client, asset, start = None, end = None, granularity = None, verbose = True, header = ''):
    """
    Returns a pandas DF with the 'Date', 'Low', 'High', 'Open', 'Close', 'Volume', 'Return', values for a given asset in the stated period and the given granularity.
    ('Date' is the imdex).    
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
    
    if asset == 'USD-USD':
        startStamp = datetime.datetime.timestamp(start)
        endStamp = datetime.datetime.timestamp(end)
        startStamp = int(startStamp/granularity)*granularity
        endStamp = int(endStamp/granularity)*granularity
        usdData = []
        while startStamp <= endStamp:
            usdData += [[datetime.datetime.fromtimestamp(startStamp),1,1,1,1,0,0]]
            startStamp += granularity
        usdDF = pd.DataFrame(usdData, columns = ['Date', 'Low', 'High', 'Open', 'Close', 'Volume','Return']).set_index('Date').sort_index().dropna()
        return usdDF
    
    if isinstance(asset, list):
        dataDict = {}
        for ass in asset:
            dataDict[ass] = getData(Client, ass, start = start, end = end, granularity = granularity, verbose = verbose, header = header + "\nGathering data for " + str(ass))
        return dataDict
    else:
        data0 = dataFromFile(asset, start = start, end = end, granularity = granularity, verbose = verbose)
    if ff.availableData(asset, start) and ff.availableData(asset, end):
        data = data0.copy()
    else:
        if len(data0) > 0:
            start = datetime.datetime.fromtimestamp(datetime.datetime.timestamp(data0.index[-1]))
        if (end - start).total_seconds()/granularity > 200:
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
                    uf.clear()
                    print(uf.progressBar(i/iMax*100, header = header + '\n'))
                candles += getCandles(Client, asset, newStart, newEnd, granularity, verbose = verbose)
                newStart = newEnd
                newEnd = newEnd + datetime.timedelta(seconds = 300*granularity)
                i += 1
            candles += getCandles(Client, asset, newEnd - datetime.timedelta(seconds = 300*granularity), end, granularity, verbose = verbose)
        else:
            uf.clear()
            print("Getting candles from Coinbase without subdivisions.")
            print(header)
            candles = getCandles(Client, asset, start, end, granularity, verbose = verbose)
        for i in range(len(candles)):
            #Converts time index
            time_in_dt = datetime.datetime.fromtimestamp(candles[i][0])
            candles[i][0] = time_in_dt

        cTimes = []
        cTemp = []
        for row in candles:
            if row[0] not in cTimes:
                cTemp += [row]
                cTimes += [row[0]]
        candles = cTemp.copy()

        candles_DF = pd.DataFrame(candles, columns = ['Date', 'Low', 'High', 'Open', 'Close', 'Volume']).set_index('Date').sort_index().dropna()
        candles_DF = ff.concatReturns(candles_DF, dropna = False)
        if len(data0) > 0 and len(candles_DF) > 0:
            candles_DF.iloc[0]['Return'] = (candles_DF.iloc[0]['Close'] - data0.iloc[-1]['Close'])/data0.iloc[-1]['Close']
        else:
            candles_DF = candles_DF
        data = data0.append(candles_DF)
    return data.drop_duplicates().dropna()

def updateData(Client, assets, granularity = 60, verbose = True, header = ''):
    for asset in assets:
        if asset != 'USD-USD':
            path = '../candlesDataBase/' + asset
            fileEnds = []
            for file in listdir(path):
                if file[-3:] == 'csv' and eval(file.split('_')[3].split('.')[0]) == granularity:
                    fileEnds += [eval(file.split('_')[2])]
            start = datetime.datetime.fromtimestamp(max(fileEnds))
            end = datetime.datetime.now()
            if (end - start).total_seconds() <= 2*granularity:
                return 0
            data = getData(Client, asset, start = start, end = end, granularity = granularity, verbose = True, header = header + "\nUpdating data for " + asset)
            startSTR = str(int(datetime.datetime.timestamp(min(data.index))))
            endSTR = str(int(datetime.datetime.timestamp(max(data.index))))
            data.reset_index(inplace = True)
            data.to_csv(path + "/" + asset + "_" + startSTR + "_" + endSTR +"_" + str(granularity) + ".csv")
    return 0
        
def getFunds(Client, dataDict, coin = 'funds'):
    #Returns value invested in each asset. buy is in t
    accs = Client.get_accounts()
    assets = [key for key in dataDict.keys()]
    funds = {asset:0 for asset in assets}
    for acc in accs:
        if acc['currency'] + "-USD" in assets:
            if coin == 'funds':
                funds[acc['currency'] + "-USD"] = eval(acc['balance'])*dataDict[acc['currency'] + "-USD"]['Close'][-1]
            elif coin == 'size':
                funds[acc['currency'] + "-USD"] = eval(acc['balance'])
    return funds

def getPortfolio(Client, dataDict):
    funds = getFunds(Client, dataDict)
    totalFunds = sum([funds[key] for key in funds.keys()])
    if totalFunds == 0:
        totalFunds = 1
    portfolio = {key:funds[key]/totalFunds for key in funds.keys()}
    return portfolio

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