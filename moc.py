from __future__ import print_function
import random
import progressbar
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys

STAR = 0
BAR = 1

def main():
    #set some constants
    nBalls = 15
    nBins = 50
    nShuffles = 10000

    #initialize stars and bars list and counts list
    starsAndBars = [STAR for _ in range(nBalls)] + [BAR for _ in range(nBins-1)]
    counts = [0 for _ in range(nBalls+1)]

    #make a progressbar for cosmetics
    pbar = progressbar.ProgressBar()

    #make nShuffles shuffles, updating counts at each one
    for shuf in pbar(range(nShuffles)):
        random.shuffle(starsAndBars)
        occMap = getOccStarsAndBars(starsAndBars,nBins)
        updateCounts(counts,occMap)
        pbar.update(shuf)

    pbar.finish()

    countsArray = np.array(counts,dtype='float64')
    logHist = np.log(countsArray/np.sum(countsArray))
    x = np.array(range(nBalls+1),dtype='float64')

    #define some convenience variables
    xOverE = x/float(nBalls)
    avN = float(nBalls)/float(nBins)
    avNInv = avN**(-1)

    #compute beta0 (macroscopic approx.) and betaX
    beta0 = np.log(1 + avNInv)
    betaX = np.log(1 + avNInv*(1 - xOverE)**(-1))

    #compute the value from theory, offset it to have the correct initial value
    boltzmann = offsetToValue(-beta0*x,logHist[0])
    theory = offsetToValue((nBalls - x)*betaX + (nBins - 2)*np.log(nBalls - x + nBins - 2),logHist[0])

    importanceSampled = np.log(importanceSamplingMain())

    plt.figure()
    plt.plot(x,logHist,'bo-',label='Sampled')
    plt.plot(x,importanceSampled,'go-',label='Importance Sampled')
    plt.plot(x,boltzmann,'k--',label='Boltzmann')
    plt.plot(x,theory,'r.-',label='Theory')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\log\,\mathrm{Pr}\,(n_1 = x)$')
    plt.legend(framealpha=0,loc='upper right')
    plt.savefig('logplot.png',dpi=400)

def importanceSamplingMain():
    #set some constants
    nBalls = 15
    nBins = 50
    nShufflesOverK = np.power(np.arange(nBalls)+1,2)*65
    totalShuffles = np.sum(nShufflesOverK)
    doneShuffles = 0

    #initialize histogram
    histogram = {'weightedCounts':np.zeros(nBalls+1),'sumWeights':0.0}

    #number of balls to hold out for importance sampling
    for k,nShuffles in zip(range(nBalls),nShufflesOverK):
        #compute importance sampling weight
        logWeight = computeImportanceSamplingWeight(nBalls,nBins-1,k)

        #initialize stars and bars array holding out the first k balls ("stacking the deck")
        starsAndBars = [STAR for _ in range(nBalls-k)]  + [BAR for _ in range(nBins-1)]

        for shuf in range(nShuffles):
            random.shuffle(starsAndBars)
            occMap = getOccStarsAndBars([STAR for _ in range(k)] + starsAndBars,nBins)
            updateHistogram(occMap,histogram,np.exp(logWeight))
            doneShuffles += 1
            if doneShuffles % 1000 == 0:
                print('on shuffle %d/%d' % (doneShuffles,totalShuffles),end='\r')

    return histogram['weightedCounts']/histogram['sumWeights']

def computeImportanceSamplingWeight(N,M,k):
    logWeight = (N + M - k)*np.log(N + M - k) + N*np.log(N) - (N - k)*np.log(N - k) - (N + M)*np.log(N + M)
    return logWeight

def offsetToValue(series,offset):
    newSeries = series - series[0]
    newSeries += offset
    return newSeries

def updateHistogram(occMap,h,weight):
    for _,v in occMap.iteritems():
        h['weightedCounts'][v] += weight
        h['sumWeights'] += weight

def updateCounts(counts,occMap):
    for _,v in occMap.iteritems():
        counts[v] += 1

def getOccStarsAndBars(starsAndBars,nBins):
    #initialize an occupancy map taking (bin number -> bin occupancy)
    occMap = {k:0 for k in range(nBins)}

    #initialize running counter variable and bar index
    counter = 0
    binIndex = 0

    #go through all the elements of the array
    for elem in starsAndBars:
        #if we find a star, just increment the counter
        if elem == STAR:
            counter += 1
        #if we find a bar...
        elif elem == BAR:
            #map the occupancy of the current bin index to the number of stars encountered
            occMap[binIndex] = counter
            #increment the bin index and set the counter to zero, we're now looking at the next bin 
            binIndex += 1
            counter = 0

    #finish up by setting the last bin's occupancy to the final counter
    occMap[binIndex+1] = counter

    return occMap

if __name__ == "__main__":
    main()
