import time
import datetime
import numpy

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
        if eta/(60*60*24) > 1:
            outString += '(approx > 1d)'
        elif eta/(60*60) > 1:
            outString += '(approx ' + time.strftime('%Hh %Mm %Ss', time.gmtime(eta)) +  ')'
        elif eta/60 > 0:
            outString += '(approx ' + time.strftime('%Mm %Ss', time.gmtime(eta)) +  ')'
        else:
            outString += '(approx ' + time.strftime('%Ss', time.gmtime(eta)) +  ')'
    return header + outString