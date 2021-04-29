#Coinbase API
import cbpro
#Packages/functions to list files in folders
from os import listdir
from os.path import isfile, join
#Clear function for jupyter
from IPython.display import clear_output
#Package for fast c-compiled functions
from numba import njit
#Package for paralell processing
from concurrent.futures import ThreadPoolExecutor

#Regular Ol' libraries
import pandas as pd
import pickle as pkl
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import time

#My Libraries
import coinbaseFunctions as cf
import utilityFunctions as uf
import financialFunctions as ff
print("shit is fixed")

#Defines and initialized Client instant to access Coinbase
with open('/home/andrerg01/AutoTraders/fredinhouTradingBot_Pvt/coinbase_credentials.pkl', 'rb') as f:
    credentials = pkl.load(f)
Client = cbpro.AuthenticatedClient(
    credentials['APIKey'],
    credentials['SecretKey'],
    credentials['passPhrase'],
    api_url = credentials['APIurl']
    )

#Assets to be considered for purchasing
assets = ['AAVE-USD', 'ADA-USD', 'ALGO-USD', 'ATOM-USD', 'BAL-USD', 'BAND-USD', 'BCH-USD', 'BNT-USD', 'BTC-USD', 'CGLD-USD', 'COMP-USD',\
                 'DASH-USD', 'EOS-USD', 'ETC-USD', 'ETH-USD', 'FIL-USD', 'GRT-USD', 'KNC-USD', 'LINK-USD', 'LRC-USD', 'LTC-USD', 'MATIC-USD',\
                 'MKR-USD', 'NMR-USD', 'NU-USD', 'OMG-USD', 'OXT-USD', 'REN-USD', 'REP-USD', 'SKL-USD', 'SNX-USD', 'SUSHI-USD', 'UMA-USD',
                 'UNI-USD','WBTC-USD', 'XLM-USD', 'XTZ-USD', 'YFI-USD', 'ZEC-USD', 'ZRX-USD']


granularity = 60*60*6
periods = 2000
end = datetime.datetime.now()
#start = end - datetime.timedelta(seconds = granularity*periods)
start = datetime.datetime(2016,3,27)

for asset in assets:
    header = 'Optimizing Strategy Parameters for ' + asset + ' at granularity ' + str(granularity) + '\n'
    cf.clear()
    print(header)
    
    header += 'Downloading asset data\n'
    cf.clear()
    print(header)
    dataOriginal = cf.getData(Client, asset, start, end, granularity = granularity, verbose = False)
    
    header += 'Calculating parameter combinations for strategy B\n'
    cf.clear()
    print(header)
    
    #Optimizing Strategy B Parameters
    RSIperiodVals = np.arange(5,int(0.1*len(dataOriginal)))
    RSILowVals = np.arange(20,49)
    RSIHighVals = np.arange(51, 80)
    #Minimum of one activation every 90 days
    minN = int(len(dataOriginal)/90)
    #Maximum of one activation per 4 days
    maxN = int(len(dataOriginal)/4)
    argCombinations = []
    for p1 in RSIperiodVals:
        for p2 in RSILowVals:
            for p3 in RSIHighVals:
                argCombinations += [(dataOriginal['close'].values, p1, p2, p3, np.array([minN, maxN], dtype = np.float64))]
    
    header += 'Calculating optimal parameters for strategy B\n'
    cf.clear()
    print(header)
    with ThreadPoolExecutor(8) as ex:   
        results = ex.map(lambda p: ff.backtestStrategyB(*p), argCombinations)
    results = np.array([result for result in results])
    
    results_df = pd.DataFrame(results, columns = ['returns', 'lengths', 'purchasedBool', 'score', 'N', 'parameters'])
    results_df = results_df.sort_values(by = 'score')
    best = results_df.tail(10).sort_values(by = 'N').iloc[-1]
    
    if best['score'] < 0:
        best['score'] = 0
        
    best = best.to_dict()
    with open(cf.dbPath + '/' + asset + '/StrBParams_'+ str(granularity) + '.pkl', 'wb') as f:
        pkl.dump(best, f)
    
    header += 'Calculating parameter combinations for strategy C\n'
    cf.clear()
    print(header)
    
    #Optimizing Strategy A Parameters
    OSCperiod1Vals = np.arange(10,40)
    OSCperiod2Vals = np.arange(20,50)
    OSCperiod3Vals = np.arange(7,20)

    OSCLowVals = np.linspace(-0.001, 0.0015, num = 10)
    OSCHighVals = np.linspace(-0.001, 0.0015, num = 10)
    #Minimum of one activation every 90 days
    minN = int(len(dataOriginal)/90)
    #Maximum of one activation per 4 days
    maxN = int(len(dataOriginal)/4)
    argCombinations = []
    for p1 in OSCperiod1Vals:
        for p2 in OSCperiod2Vals:
            for p3 in OSCperiod3Vals:
                for p4 in OSCLowVals:
                    for p5 in OSCHighVals:
                        if p2 > p1 > p3 and p5 > p4:
                            argCombinations += [(dataOriginal['close'].values, p1, p2, p3, p4, p5, np.array([minN, maxN], dtype = np.float64))]
                            
    header += 'Calculating optimal parameters for strategy C\n'
    cf.clear()
    print(header)                        
    with ThreadPoolExecutor(8) as ex:   
        results = ex.map(lambda p: ff.backtestStrategyC(*p), argCombinations)
    results = np.array([result for result in results])
    
    results_df = pd.DataFrame(results, columns = ['returns', 'lengths', 'purchasedBool', 'score', 'N', 'parameters'])
    results_df = results_df.sort_values(by = 'score')
    best = results_df.tail(10).sort_values(by = 'N').iloc[-1]
    if best['score'] < 0:
        best['score'] = 0
    best = best.to_dict()
    with open(cf.dbPath + '/' + asset + '/StrCParams_'+ str(granularity) + '.pkl', 'wb') as f:
        pkl.dump(best, f)
    
    