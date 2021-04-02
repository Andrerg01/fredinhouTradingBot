import cbpro
import pandas as pd
import pickle as pkl
import datetime
import matplotlib.pyplot as plt
import financialFunctions as ff
import coinbaseFunctions as cf
import utilityFunctions as uf
import numpy as np
import time
import matplotlib.patches as patches
import math
from decimal import *

defaultAssets = ['LRC-USD', 'LTC-USD', 'MATIC-USD', 'MKR-USD', 'NMR-USD', 'NU-USD', 'OMG-USD', 'OXT-USD', 'REN-USD', 'REP-USD', 'SKL-USD', 'SNX-USD', 'SUSHI-USD', 'UMA-USD', 'UNI-USD','WBTC-USD', 'XLM-USD', 'XTZ-USD', 'YFI-USD', 'ZEC-USD', 'ZRX-USD']

loopTime = datetime.timedelta(hours = 1)

assets = defaultAssets.copy()
with open('../../../fredinhouTradingBot_Pvt/coinbase_credentials_1.pkl', 'rb') as f:
    credentials = pkl.load(f)
Client = cbpro.AuthenticatedClient(
    credentials['APIKey'],
    credentials['SecretKey'],
    credentials['passPhrase'],
    api_url = credentials['APIurl']
    )
del credentials
end = datetime.datetime.now()
start = end - datetime.timedelta(days = 5*365)

i = 1
time0 = datetime.datetime.now()
for asset in assets:
    startTemp = start
    endTemp = start + datetime.timedelta(days = 30)
    time02 = datetime.datetime.now()
    pct1 = i/len(assets)*100
    pb1 = uf.progressBar(pct1, time0 = time0, header = "Downloading " + asset + " data\n") + "\n"
    while endTemp < end - datetime.timedelta(days = 30):
        pct2 = (endTemp - start).total_seconds()/(end - start).total_seconds()*100
        pb2 = uf.progressBar(pct2, time0 = time02, header = pb1)
        uf.clear()
        print(pb2)
        candles = cf.getData(Client, asset, start = startTemp, end = endTemp, granularity = 60, verbose = False)
        
        with open("../candlesDataBase/" + asset + "/" + asset + "_" + str(int(datetime.datetime.timestamp(startTemp))) + "_" + str(int(datetime.datetime.timestamp(endTemp))) + "_60.pkl", 'wb') as f:
            pkl.dump(candles, f)
            
        startTemp = endTemp
        endTemp = endTemp + datetime.timedelta(days = 30)
    
    i += 1
    
    candles = cf.getData(Client, asset, startTemp, end, 60, verbose = True)
    with open("../candlesDataBase/" + asset + "/" + asset + "_" + str(int(datetime.datetime.timestamp(startTemp))) + "_" + str(int(datetime.datetime.timestamp(end))) + "_60.pkl", 'wb') as f:
        pkl.dump(candles, f)