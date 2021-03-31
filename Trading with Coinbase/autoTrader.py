import cbpro
import pandas as pd
import pickle as pkl
import datetime
import matplotlib.pyplot as plt
import financialFunctions as ff
import numpy as np
import time
import matplotlib.patches as patches
import math
from decimal import *

defaultAssets = ['AAVE-USD', 'ADA-USD', 'ALGO-USD', 'ATOM-USD', 'BAL-USD', 'BAND-USD', 'BCH-USD', 'BNT-USD', 'BTC-USD', 'CGLD-USD', 'COMP-USD', 'DASH-USD', 'EOS-USD', 'ETC-USD', 'ETH-USD', 'FIL-USD', 'GRT-USD', 'KNC-USD', 'LINK-USD', 'LRC-USD', 'LTC-USD', 'MATIC-USD', 'MKR-USD', 'NMR-USD', 'NU-USD', 'OMG-USD', 'OXT-USD', 'REN-USD', 'REP-USD', 'SKL-USD', 'SNX-USD', 'SUSHI-USD', 'UMA-USD', 'UNI-USD','WBTC-USD', 'XLM-USD', 'XTZ-USD', 'YFI-USD', 'ZEC-USD', 'ZRX-USD']

loopTime = datetime.timedelta(hours = 1)

while(True):
    with open("command.txt") as f:
        command = f.read()
    if command == 'stop':
        exit()
    assets = defaultAssets.copy()
    with open('../../fredinhouTradingBot_Pvt/coinbase_credentials.pkl', 'rb') as f:
        credentials = pkl.load(f)
    Client = cbpro.AuthenticatedClient(
        credentials['APIKey'],
        credentials['SecretKey'],
        credentials['passPhrase'],
        api_url = credentials['APIurl']
        )
    del credentials

    timeStart = datetime.datetime.now()
    header = "Initializing Client.\n"
    print(header)
    with open('../../fredinhouTradingBot_Pvt/coinbase_credentials.pkl', 'rb') as f:
        credentials = pkl.load(f)
    Client = cbpro.AuthenticatedClient(
        credentials['APIKey'],
        credentials['SecretKey'],
        credentials['passPhrase'],
        api_url = credentials['APIurl']
        )
    del credentials
    
    header += "Done!\n"
    header += "Gathering asset data.\n"
    ff.clear()
    print(header)
    time.sleep(1)
    
    end = datetime.datetime.now() + datetime.timedelta(hours = 5)
    start = end - datetime.timedelta(days = 30)
    price_df = ff.getClosingPrice(Client, asset = assets, start = start, end = end, granularity = 60, verbose = True, usd = True, header = header)
    
    header += "Done!\n"
    header += "Calculating returns.\n"
    ff.clear()
    print(header)
    time.sleep(1)
    
    returns_df = ff.getReturns(Client, price_df = price_df)
    
    assets = sorted(assets + ['USD-USD'])
    
    header += "Done!\n"
    header += "Retrieving funds and portfolio.\n"
    ff.clear()
    print(header)
    time.sleep(1)
    
    funds = ff.getFunds(Client, price_df)
    
    totalFunds = sum([funds[key] for key in funds.keys()])
    currentPortfolio = ff.getPortfolio(Client, price_df)
        
    header += "Done!\n"
    header += "Current Funds: " + str(totalFunds) + " USD \nCurrent Portfolio: " + ff.printPortfolio(currentPortfolio) + ".\n"
    ff.clear()
    print(header)
    time.sleep(1)
        
    header += "Intervalizing Returns.\n"
    
    ff.clear()
    print(header)
    time.sleep(1)
    intReturns = ff.intervalizeReturns(returns_df, interval = 'W')
    intReturns = pd.DataFrame(intReturns.values, index = intReturns.index.values, columns = ['w-Returns']).sort_values('w-Returns')
    header += "Done!\n"
    header += "Finding the good assets.\n"
    ff.clear()
    print(header)
    time.sleep(1)
    goodAssets = []
    for i in range(len(intReturns)):
        if intReturns.iloc[i]['w-Returns'] > 0:
            goodAssets += [intReturns.index[i]]
            
    header += "Done!\n"
    header += 'Good assets: '
    for asset in goodAssets:
        header += asset + ":" + str(intReturns['w-Returns'][asset]) + " | "
    header = header[:-3] + "\n"
    
    header += 'Done!\n'
    header += "Making plot and calculating optimal portfolio.\n"
    ff.clear()
    print(header)
    time.sleep(1)
    
    fig, ax = plt.subplots(2, 2,figsize = [15, 15/1.61])
    #calculating efficiency frontier
    minR = min(intReturns['w-Returns'][goodAssets])
    maxR = max(intReturns['w-Returns'][goodAssets])

    
    EFxs = []
    EFys = []
    for target_return in np.linspace(minR, maxR, 50):
        EFws = ff.minimizeVolatility(target_return, returns_df[goodAssets], interval = 'W')
        EFxs += [ff.portfolioVolatility(returns_df[goodAssets], EFws)]
        EFys += [ff.portfolioReturn(returns_df[goodAssets], EFws, interval = 'W')]
        
    EFxs = np.array(EFxs)
    EFys = np.array(EFys)
    ax[0, 0].plot(EFxs, EFys, marker = '.', color = 'midnightblue', label = 'Efficiency Frontier')

    goodAssets = sorted(goodAssets + ['USD-USD'])
    
    currAssets = [key for key in currentPortfolio.keys()]
    currW = np.array([currentPortfolio[key] for key in currAssets])
    currX = ff.portfolioVolatility(returns_df[currAssets], currW)
    currY = ff.portfolioReturn(returns_df[currAssets], currW, interval = 'W')
    ax[0, 0].scatter(currX, currY, marker = '.', color = 'green', label = 'Current Portfolio')
    
    msrW = ff.maximumSharpeRatio(0.0, returns_df[goodAssets], interval = 'W')
    msrY = ff.portfolioReturn(returns_df[goodAssets], msrW, interval = 'W')
    msrX = ff.portfolioVolatility(returns_df[goodAssets], msrW)
   
    ax[0, 0].scatter(msrX, msrY, color = 'red', marker = 'o', label = 'Maximum Sharpe Ratio Portfolio')
    ax[0, 0].scatter(0, 0, marker = 'o', color = 'red')
    ax[0, 0].plot([0,2*msrX],[0, 2*msrY],'--',color = 'red')
    
    msrPortfolio = {goodAssets[i]:msrW[i] for i in range(len(goodAssets))}
    
    nicePort = ff.nicefyPortfolio(msrPortfolio, treshold = 0.05)
    niceW = np.array([nicePort[key] for key in nicePort.keys()])
    
    for asset in assets:
        if asset not in nicePort.keys():
            nicePort[asset] = 0.0
            
    niceWY = ff.portfolioReturn(returns_df[goodAssets], niceW, interval = 'W')
    niceWX = ff.portfolioVolatility(returns_df[goodAssets], niceW)
   
    ax[0, 0].scatter(niceWX, niceWY, marker = 'o', color = 'orange', label = "Nicefied MSR Portfolio")
    
    factor = 0.05
    xrange = max(EFxs) - min(EFxs)
    yrange = max(EFys) - min(EFys)
    acceptable_range = patches.Ellipse((niceWX, niceWY), xrange*factor, yrange*factor, color='blue', alpha = .35)
    ax[0, 0].add_patch(acceptable_range)
    
    ax[0, 0].set_xlim(0, max(EFxs))
    ax[0, 0].set_ylim(0, maxR)
    
    ax[0, 0].set_xlabel("Volatility")
    ax[0, 0].set_ylabel("Weekly Return")
    ax[0, 0].set_title("Portfolio performances plot.")
    ax[0, 0].legend()
    
    
    header += 'Done!\n'
    header += "Nicefied \"Best\" portfolio: " + ff.printPortfolio(nicePort) + "\n"
    header += "Return: " + str(niceWY) + "\n"
    header += "Volatility: " + str(niceWX) + "\n"
    ff.clear()
    print(header)
    
    buys = {}
    sells = {}
 
    header += "Calculating trades.\n"
    ff.clear()
    print(header)
    time.sleep(1)

    porA = currentPortfolio.copy()
    porB = nicePort.copy()

    for key in porA.keys():
        if porB[key] > porA[key]:
            buys[key] = porB[key] - porA[key]
        elif porB[key] < porA[key]:
            sells[key] = porA[key] - porB[key]
    buys, sells = ff.nicefyTrades(buys, sells, 0.05)
    header += "Purchases to be made (in amount of asset):   "
    for key in buys.keys():
        buys[key] = buys[key]*totalFunds/(price_df[key][-1])
        header += key + ":" + str(buys[key]) + " | "
    header = header[:-3] + "\nSales to be made (in amount of asset):   "
    for key in sells.keys():
        sells[key] = sells[key]*totalFunds/(price_df[key][-1])
        header += key + ":" + str(sells[key]) + " | "
    header = header[:-3] + "\n"
    header += "Done!\n"
    header += "Making necessary trades!\n"
    ff.clear()
    print(header)
    time.sleep(1)

    ff.makeTrades(Client, buys, sells)

    if len(buys) == 0 and len(sells) == 0:
        rebalance = False
    else:
        rebalance = True
        currentPortfolio = ff.getPortfolio(Client, price_df)
        funds = ff.getFunds(Client, price_df)
        totalFunds = sum([funds[key] for key in funds.keys()])
        

    header += "Done!\n"
        
    
    with open('fundsHist.pkl', 'rb') as f:
        fundsHist = pkl.load(f)
    fundsHist = fundsHist.append(pd.DataFrame({'Date':[datetime.datetime.now()], 'Funds':[totalFunds], 'Portfolio':[currentPortfolio], 'Rebalance':[rebalance], 'Buys':[buys], 'Sells':[sells]}).set_index('Date').sort_index())
    
    ax[0, 1].plot(fundsHist['Funds'])
    ax[0, 1].set_xlabel('Date')
    ax[0, 1].set_ylabel('Total Funds (USD)')
    ax[0, 1].set_title('Total Funds History')
    #ax[0, 1].set_xlim(datetime.datetime.now() - datetime.timedelta(months = 1), datetime.datetime.now())
    for i in range(len(fundsHist)):
        if fundsHist.iloc[i]['Rebalance']:
            ax[0, 1].plot([fundsHist.index.values[i], fundsHist.index.values[i]], [min(fundsHist['Funds']), max(fundsHist['Funds'])], '--', color = 'red')
    returnsHist = fundsHist['Funds'].pct_change().dropna()
    axTemp = ax[0, 1].twinx()
    axTemp.bar(returnsHist[returnsHist > 0].index, returnsHist[returnsHist > 0], color = 'green', width = (fundsHist.index.values[-1] - fundsHist.index.values[-2])/3) 
    axTemp.bar(returnsHist[returnsHist <= 0].index, abs(returnsHist[returnsHist <= 0]), color = 'red', width = (fundsHist.index.values[-1] - fundsHist.index.values[-2])/3) 
    axTemp.set_ylim(0, 5*max(returnsHist))
    axTemp.set_ylabel("Return")
    everInvestedAssets = []
    for port in fundsHist['Portfolio']:
        for key in port.keys():
            if port[key] > 0 and key not in everInvestedAssets:
                everInvestedAssets += [key]
    assetInvestOverTime = {asset:[] for asset in everInvestedAssets}
    for port in fundsHist['Portfolio']:
        for asset in everInvestedAssets:
            assetInvestOverTime[asset] += [port[asset]]
    for asset in everInvestedAssets:
        ax[1, 0].plot(fundsHist.index, assetInvestOverTime[asset], label = asset)
    ax[1, 0].legend(title='Asset', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[1, 0].set_title('Asset share in portfolio over time.')
    ax[1, 0].set_xlabel("Date")
    ax[1, 0].set_ylabel("Portfolio share")

    heights = [fundsHist.iloc[-1]['Portfolio'][asset] for asset in everInvestedAssets]
    ax[1, 1].bar([asset[:3] for asset in everInvestedAssets], heights)
    for tick in ax[1, 1].get_xticklabels():
        tick.set_rotation(90)
    ax[1, 1].set_xlabel("Asset")
    ax[1, 1].set_ylabel("Share in current portfolio")
    ax[1, 1].set_title("Layout of current portfolio")
    
    with open('fundsHist.pkl', 'wb') as f:
        pkl.dump(fundsHist, f)
    
    ff.clear()
    print(header)
    
    fig.tight_layout()
    fig.savefig("PortfolioPerformances.png")
    timeEnd = datetime.datetime.now()
    timeEllapsed = timeEnd - timeStart
    
    if timeEllapsed >= loopTime:
        timeSleep = 0
    else:
        timeSleep = int((loopTime - timeEllapsed).total_seconds())
        
    print("Checking for more data at: " + str(datetime.datetime.now() + (loopTime - timeEllapsed)))
    time.sleep(timeSleep)
    
    