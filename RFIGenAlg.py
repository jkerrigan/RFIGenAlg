import numpy as np
import pyuvdata
import pylab as pl
import scipy.fftpack as sfft

class rfiGenAlg:
    
    def __init__(self,data,random_crossover = False, initWalker = None):
        self.data = data
        self.walkerArray = []
        self.walkerScore = []
        self.epochScore = []
        self.random_crossover = random_crossover
        self.avgFitness = None
        self.growthVolArray = []
        self.mutRateArray = []
        self.initWalker = initWalker
        
    def addWalker(self,initWalker=None):
        if initWalker == None:
            self.walkerArray.append(self.mutate(np.ones(self.data.shape))) #self.randomRFIMap())
        else:
            #if np.random.rand()<0.2: Replaced with an adaptive mutation rate
            try:
                mutationRate = self.epochScore[-1]/(1.*self.avgFitness) #np.abs(1 - self.epochScore[-1]/1.*self.avgFitness))
            except:
                mutationRate = 1.
            self.mutRateArray.append(mutationRate)
            try:
                if np.var(self.epochScore) > np.abs(np.mean(self.epochScore) - self.epochScore[-1]):#np.random.rand() < mutationRate:
                    try:
                        growthVolatility = np.var(self.epochScore[:50])/np.var(self.epochScore[-50:])
                    except:
                        growthVolatility = 0.01
                    self.growthVolArray.append(growthVolatility)
                    if np.random.rand() > 0.4:#growthVolatility:
                        self.walkerArray.append(self.mutate(initWalker))
                    else:
                    ### Grow mutations instead of introducing new mutations
                        self.walkerArray.append(self.growMutations(initWalker))
                else:
                    self.walkerArray.append(initWalker)
            except:
                self.walkerArray.append(initWalker)
        
    def randomRFIMap(self):
        rfiMap = np.random.randint(0,2,size=np.shape(self.data)).astype(bool)
        return rfiMap
    
    def delTrans(self,walker_to_score):
        WA_ = sfft.fftshift(sfft.fft(self.data*self.walkerArray[walker_to_score],axis=1),axes=1)
        #WA_ = sfft.fftshift(sfft.fft(self.data*self.walkerArray[walker_to_score],axis=0),axes=0)
        #_WA = np.fft.fftshift(np.fft.fft(WA_,axis=0),axes=0)
        ma_WA = self.mask()*WA_
        return np.sum(np.abs(ma_WA))
        
    def mask(self):
        hole = np.ones_like(self.data)
        sh = np.shape(hole)
        hole[:,sh[1]/2 - 20:sh[1]/2 + 20] = 0.
        ## best results for HERA were 20
        #hole[sh[0]/2 - 5:sh[0]/2 +5,:] = 0
        #for i in range(-5,5):
        #    for j in range(-20,20):
        #        hole[sh[0]/2 + i,sh[1]/2 + j] = 0.
        return hole
        
    def score(self,walker_to_score):
        delfringeScore = self.delTrans(walker_to_score)
        ### Try multiple fitness functions: del transform and # of flags
        self.walkerScore.append(delfringeScore + np.sum(np.logical_not(self.initWalker)))
        #print delfringeScore
        
    
    def plotRFIMap(self,live=False):
        pl.imshow(self.initWalker,aspect='auto',interpolation='none')
        if live:
            pl.show()
        
    def mutate(self,rfiMap):
        #if np.random.rand()>0.5:
        #    walker = rfiMap.ravel(order='A')
        #else:
        #    walker = rfiMap.flatten(order='F')
        sh = np.shape(rfiMap)
        mutations = np.random.randint(0,100)
        rfiMapC = np.copy(rfiMap)
        for i in range(mutations):
            if np.random.rand()>0.0: # I turned on minimal RFI in frequency mutations
                mut_freq = np.random.randint(0,sh[1])
                mut_time_a = np.random.randint(0,sh[0]+1)
                mut_time_b = np.random.randint(0,sh[0]+1)
                if mut_time_a>mut_time_b:
                    if np.random.rand()>0.1:
                        rfiMapC[mut_time_b:mut_time_a,mut_freq] = np.ones(mut_time_a-mut_time_b)
                    else:
                        rfiMapC[mut_time_b:mut_time_a,mut_freq] = np.zeros(mut_time_a-mut_time_b)
                else:
                    if np.random.rand()>0.1:
                        rfiMapC[mut_time_a:mut_time_b,mut_freq] = np.ones(mut_time_b-mut_time_a)
                    else:
                        rfiMapC[mut_time_a:mut_time_b,mut_freq] = np.zeros(mut_time_b-mut_time_a)
            else:
                mut_freq_a = np.random.randint(0,sh[1]+1)
                mut_freq_b = np.random.randint(0,sh[1]+1)
                mut_time = np.random.randint(0,sh[0])
                if mut_freq_a>mut_freq_b:
                    if np.random.rand()>0.1:
                        rfiMapC[mut_time,mut_freq_b:mut_freq_a] = np.ones(mut_freq_a-mut_freq_b)
                    else:
                        rfiMapC[mut_time,mut_freq_b:mut_freq_a] = np.zeros(mut_freq_a-mut_freq_b)
                else:
                    if np.random.rand()>0.1:
                        rfiMapC[mut_time,mut_freq_a:mut_freq_b] = np.ones(mut_freq_b-mut_freq_a)
                    else:
                        rfiMapC[mut_time,mut_freq_a:mut_freq_b] = np.zeros(mut_freq_b-mut_freq_a)                        
                        
        #mut_a = np.random.randint(0,len(walker))
        #mut_b = np.random.randint(0,len(walker))
        #if np.random.rand()>0.5:
        #    mutation = np.random.randint(0,2,size=np.abs(mut_b-mut_a))
        #else:
        #    mutation = np.ones(np.abs(mut_b-mut_a))
        #if mut_a > mut_b:
        #    walker[mut_b:mut_a] = mutation
        #else:
        #    walker[mut_a:mut_b] = mutation
        return rfiMapC #walker.reshape(np.shape(self.data))

    def growMutations(self,rfiMap):
        #sh = np.shape(rfiMap)
        zeros = np.array(np.where(rfiMap==0))
        sh = np.shape(zeros)
        rfiMapC = np.copy(rfiMap)
        #growthSites = np.random.randint(1,100)
        #for growth in range(growthSites):
            #i = np.random.randint(0,sh[0])
            #j = np.random.randint(0,sh[1])
        i = zeros[0,:]
        j = zeros[1,:]
#        if rfiMap[i,j] == 0:
        try:
            rfiMapC[i,j+1] = np.random.randint(0,2)
            rfiMapC[i+1,j] = np.random.randint(0,2)
            rfiMapC[i+1,j+1] = np.random.randint(0,2)
            rfiMapC[i,j-1] = np.random.randint(0,2)
            rfiMapC[i-1,j] = np.random.randint(0,2)
            rfiMapC[i-1,j-1] = np.random.randint(0,2)
        except:
            pass
        return rfiMapC

        
    def resetWalkers(self):
        self.walkerArray = []
        self.walkerScore = []
    
    def findBestWalker(self):
        #if np.argmin(np.sum(np.sum(self.walkerArray,1),1)) == np.argmin(self.walkerScore):
        # Introduce random mating of top 2 scorers of population in frequency
        offspring = np.zeros_like(self.data)
        top1 = self.walkerArray[np.argmin(self.walkerScore)]
        secondBestIndx = np.argwhere(self.walkerScore==np.sort(self.walkerScore)[1])
        thirdBestIndx = np.argwhere(self.walkerScore==np.sort(self.walkerScore)[2])
        #print len(secondBestIndx),secondBestIndx
        try:
            top2 = self.walkerArray[secondBestIndx[0][0]]
            top3 = self.walkerArray[thirdBestIndx[0][0]]
        except:
            top2 = self.walkerArray[secondBestIndx[0]]
            top3 = self.walkerArray[thirdBestIndx[0]]
        if self.random_crossover:
            matingChain = np.random.randint(0,self.data.shape[1],size=self.data.shape[1]/np.random.randint(1,5))
            matingChain2 = np.random.randint(0,self.data.shape[1],size=self.data.shape[1]/np.random.randint(1,10))
        else:
            matingChain = np.random.randint(0,self.data.shape[1],size=self.data.shape[1]/2)
            matingChain2 = np.random.randint(0,self.data.shape[1],size=self.data.shape[1]/2)
        top1[:,matingChain] = top2[:,matingChain]
        top1[:,matingChain2] = top3[:,matingChain2]
        self.initWalker = top1 #self.walkerArray[np.argmin(self.walkerScore)]
        self.epochScore.append(self.walkerScore[np.argmin(self.walkerScore)])
        self.avgFitness = np.mean(self.walkerScore)
        #else:
        #    self.epochScore.append(self.walkerScore[np.argmin(self.walkerScore)])

        #print np.argmin(self.walkerScore),np.argmax(np.sum(np.sum(self.walkerArray,1),1))
    
    def livePlot(self):
        self.plot.set_data(np.log10(np.abs(self.data*self.initWalker)))
        pl.draw()
        pl.show()
        pl.pause(0.01)



 
    def runEpoch(self,num_of_walkers,epoch):
        pl.ion()
        for i in range(num_of_walkers):
            if self.initWalker != None:
                self.addWalker(self.initWalker)
            else:
                self.addWalker()
                #self.addWalker(self.initWalker)
            self.score(i)
        self.findBestWalker()
        self.resetWalkers()        
        if epoch == 0:
            self.plot = pl.imshow(np.log10(np.abs(self.data*self.initWalker)),aspect='auto',interpolation='none')
        self.livePlot()

        
        
