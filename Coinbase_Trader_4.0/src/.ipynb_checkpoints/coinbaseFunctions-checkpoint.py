import numpy as np
import pandas as pd
import time
import cbpro
import os
from os import listdir
from os.path import isfile, join
import datetime
import utilityFunctions as uf
#Remove when not prototyping
from IPython.display import clear_output

dbPath = '/home/andrerg01/AutoTraders/DataBase/Crypto'

def clear():
    os.system('clear')
    #clear_output()
    
def getCandles(Client, asset, start, end, granularity, triesMax = 10, verbose = True, header = ''):
    #Function to get candles data straight from coinbase for a single asset
    #Loops for 10 tries or until we get the data successfully.
    if verbose: 
        header += "Retrieving data for " + asset + "\n"
        clear()
        print(header)
        
    #If there are more than 300 indexes to be imported, it will subdivide the request automatically
    if (end - start).total_seconds()/granularity > 300:
        if verbose: 
            header += "Granularity too small, subdividing requests.\n"
            clear()
            print(header)
        #Temporary start and end for this portion of the data.
        newStart = start
        newEnd = start + datetime.timedelta(seconds = 300*granularity)
        candles = []
        i = 1
        iMax = int((end - start).total_seconds()/(300*granularity)) + 1
        while newEnd < end:
            if verbose:
                clear()
                print(uf.progressBar(i/iMax*100, header = header + '\n'))
            candles += getCandles(Client, asset, newStart, newEnd, granularity, verbose = verbose)
            newStart = newEnd
            newEnd = newEnd + datetime.timedelta(seconds = 300*granularity)
            i += 1
        candles += getCandles(Client, asset, newEnd - datetime.timedelta(seconds = 300*granularity), end, granularity, verbose = verbose)
    else:
        tries = 0
        success = False
        while tries < triesMax and not success:
            #standard function to call for historic data
            candles = Client.get_product_historic_rates(asset, start = start.isoformat(), end = end.isoformat(), granularity = granularity)
            #candles will be 1 unit long if it is an error message
            if not isinstance(candles, dict):
                success = True
            else:
                if verbose: 
                    header += "Failure in retreiving data for " + asset + ". Waiting 1 second and trying again. " + str(tries + 1) + "/" + str(triesMax) + "\n" + str(candles)
                    clear()
                    print(header)
                tries += 1
                time.sleep(.1)
        if not success:
            print("Error getting candles for " + asset)
            print(candles)
            return []
    clear()
    return candles

def dataFullInFileQ(asset, start, end, granularity):
    path = dbPath + "/" + asset
    allFiles = [f for f in listdir(path) if isfile(join(path, f))]
    starts = []
    ends = []
    for file in allFiles:
        if file.endswith("-" + str(granularity) + ".csv"):
            starts += [datetime.datetime.fromtimestamp(eval(file.split('-')[0]))]
            ends += [datetime.datetime.fromtimestamp(eval(file.split('-')[1]))]
    if len(starts) == 0:
        return False
    if min(starts) <= start < max(ends) and min(starts) < end <= max(ends):
        return True
    else:
        return False
            
def dataPartInFileQ(asset, start, end, granularity):
    path = dbPath + "/" + asset
    allFiles = [f for f in listdir(path) if isfile(join(path, f))]
    starts = []
    ends = []
    for file in allFiles:
        if file.endswith("-" + str(granularity) + ".csv"):
            starts += [datetime.datetime.fromtimestamp(eval(file.split('-')[0]))]
            ends += [datetime.datetime.fromtimestamp(eval(file.split('-')[1]))]
    if len(starts) == 0:
        return False
    if start <= max(ends) < end:
        return True
    else:
        return False

def getData(Client, asset, start, end, granularity, triesMax = 10, verbose = True, header = ''):
    #First check if data is fully in file, if so, it is gathered from there.
    if dataFullInFileQ(asset, start, end, granularity):
        if verbose:
            header += "Data fully contained in local database.\n"
            clear()
            print(header)
        path = dbPath + "/" + asset
        allFiles = [f for f in listdir(path) if isfile(join(path, f))]
        goodFiles = []
        starts = []
        ends = []
        for file in allFiles:
            if file.endswith("-" + str(granularity) + ".csv"):
                goodFiles += [file]
                starts += [datetime.datetime.fromtimestamp(eval(file.split('-')[0]))]
                ends += [datetime.datetime.fromtimestamp(eval(file.split('-')[1]))]
        goodFilesStart = []
        goodFilesEnd = []
        for i in range(len(goodFiles)):
            if starts[i] <= start:
                goodFilesStart += [goodFiles[i]]
            if end <= ends[i]:
                goodFilesEnd += [goodFiles[i]]
        goodFiles = list(set(goodFilesStart).intersection(set(goodFilesEnd)))
        data = pd.read_csv(path + "/" + goodFiles[0], parse_dates=['date'])
        for i in range(1, len(goodFiles)):
            data = data.append(pd.read_csv(path + "/" + goodFiles[i], parse_dates=['date']))
        data = data.set_index("date").sort_index()[start:end]
    elif dataPartInFileQ(asset, start, end, granularity):
        if verbose:
            header += "Data partially contained in local database.\n"
            clear()
            print(header)
        path = dbPath + "/" + asset
        allFiles = [f for f in listdir(path) if isfile(join(path, f))]
        goodFiles = []
        starts = []
        ends = []
        for file in allFiles:
            if file.endswith("-" + str(granularity) + ".csv"):
                goodFiles += [file]
                starts += [datetime.datetime.fromtimestamp(eval(file.split('-')[0]))]
        greatFiles = []
        for i in range(len(goodFiles)):
            if starts[i] <= start:
                greatFiles += [goodFiles[i]]
        data = pd.read_csv(path + "/" + goodFiles[0], parse_dates=['date'])
        for i in range(1, len(goodFiles)):
            data = data.append(pd.read_csv(path + "/" + goodFiles[i], parse_dates=['date']))
        data = data.set_index("date").sort_index()[start:]
        data = data.append(getData(Client, asset, data.index[-1] + datetime.timedelta(seconds = granularity), end, granularity, triesMax = 10, verbose = True, header = ''))
    else:
        #Retrieving Candles
        candles = getCandles(Client, asset, start, end, granularity, triesMax = triesMax, verbose = verbose, header = header)
        #Changes date from timestamp to datetime
        for i in range(len(candles)):
            #Converts time index
            time_in_dt = datetime.datetime.fromtimestamp(candles[i][0])
            candles[i][0] = time_in_dt
        candles = np.array(candles)
        cols = ['date', 'low', 'high', 'open', 'close', 'volume']
        candles = {cols[j]:[candles[i][j] for i in range(len(candles))] for j in range(len(cols))}
        data = pd.DataFrame(candles).set_index('date').sort_index()
    return data.drop_duplicates()

def updateData(Client, asset, granularity, verbose = True):
    start = datetime.datetime(2016,3,27)
    end = datetime.datetime.now()
    data = getData(Client, asset, start, end, granularity = granularity, verbose = verbose)
    start = data.index[0]
    path = dbPath + "/" + asset
    while start < end:
        endTemp = start + datetime.timedelta(seconds = granularity*2**12)
        dataTemp = data[start:endTemp].iloc[:-1]
        startID = int(datetime.datetime.timestamp(dataTemp.index[0]))
        endTempID = int(datetime.datetime.timestamp(dataTemp.index[-1]))
        dataTemp.to_csv(path + "/" + str(startID) + "-" + str(endTempID) + "-" + str(granularity) + ".csv")
        start = start + datetime.timedelta(seconds = granularity*2**12)
        
def getFunds(Client, assets, allCloseData):
    #Returns value invested in each asset. buy is in t
    dataDict = {assets[i]:allCloseData[i] for i in range(len(assets))}
    accs = Client.get_accounts()
    funds = {asset:0 for asset in assets}
    for acc in accs:
        if acc['currency'] + "-USD" in (assets + ['USD-USD']):
            if acc['currency'] + "-USD" == 'USD-USD':
                funds[acc['currency'] + "-USD"] = eval(acc['balance'])
            else:
                funds[acc['currency'] + "-USD"] = eval(acc['balance'])*dataDict[acc['currency'] + "-USD"][-1]
    return funds

def getPortfolio(Client, assets, allCloseData):
    funds = getFunds(Client, assets, allCloseData)
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
                time.sleep(1)
                if len(sellOrder) > 1:
                    placed = True
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
                    elif sellOrder['message'].startswith("size is too small.") or sellOrder['message'].startswith("size must be a number.") or sellOrder['message'].startswith("Limit only mode."):
                        pct = 0
                        print("Too tired for this crap")
                    else:    
                        pct -= 0.05
                        size = sells[key]*pct
                        print("Order placement not successfull, trying again with " + str(pct*100) + "% of original amount in 1 second")
#             if placed:
#                 while not completed:
#                     try:
#                         orderStatus = Client.get_order(sellOrder['id'])['status']
#                     except:
#                         orderStatus = 'Error'
#                     time.sleep(1)
#                     if orderStatus == 'done':
#                         print("Order to sell " + str(size) + " " + key + " has been successfully completed.")
#                         completed = True
#                     else:
#                         print("Order did not yet complete, waiting 1 second and trying again.") 
    for key in buys.keys():
        if key != 'USD-USD':
            placed = False
            completed = False
            pct = 1.0
            size = buys[key]*pct
            while not placed and pct > 0:
                print("Placing an order to buy " + str(size) + " " + key)
                buyOrder = Client.place_market_order(product_id = key, side = 'buy', size = size)
                time.sleep(1)
                if len(buyOrder) > 1:
                    placed = True
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
                    elif buyOrder['message'].startswith("size is too small.") or buyOrder['message'].startswith("size must be a number.") or buyOrder['message'].startswith("Limit only mode"):
                        pct = 0
                    else:    
                        pct -= 0.05
                        size = buys[key]*pct
                        print("Order placement not successfull, trying again with " + str(pct*100) + "% of original amount.")
#             if placed:
#                 while not completed:
#                     try:
#                         orderStatus = Client.get_order(sellOrder['id'])['status']
#                     except:
#                         orderStatus = 'Error'
#                     time.sleep(1)
#                     if orderStatus == 'done':
#                         print("Order to buy " + str(size) + " " + key + " has been successfully completed.")
#                         completed = True
#                     else:
#                         print("Order did not yet complete, waiting 1 second and trying again.")
#                         print(orderStatus)    
