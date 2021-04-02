import os
import datetime
import numpy as np
import time
import pickle as pkl
from IPython.display import clear_output

def clear():
    try:
        get_ipython().__class__.__name__
        clear_output()
    except:
        os.system('clear')
        
def progressBar(pct, time0 = 0, header = ''):
    time1 = datetime.datetime.now()
    if time0 == 0:
        eta = ''
    else:
        deltaT = (time1-time0).total_seconds()
        try:
            eta = 100.*deltaT/pct - deltaT
        except:
            eta = ''
    numberOfHashtags = int(pct/2) + 1
    numberOfdots = 50-numberOfHashtags
    outString = "[" + "#"*numberOfHashtags + '.'*numberOfdots + '] '
    if eta != '':
        #outString += '(approx ' + '%0.2f' % eta + ' seconds)'
        if eta/(60*60*24) > 1:
            outString += '(approx > 1d)'
        elif eta/(60*60) > 1:
            outString += '(approx ' + time.strftime('%Hh %Mm %Ss', time.gmtime(eta)) +  ')'
        elif eta/60 > 0:
            outString += '(approx ' + time.strftime('%Mm %Ss', time.gmtime(eta)) +  ')'
        else:
            outString += '(approx ' + time.strftime('%Ss', time.gmtime(eta)) +  ')'
    return header + outString

def isNone(var):
    #Check if a variable == None easily
    try:
        len(var)
        return False
    except:
        return True
 