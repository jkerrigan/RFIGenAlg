import numpy as np
import pyuvdata
import pylab as pl
import scipy.fftpack as sfft

class rfiGenAlg:
    
    def __init__(self,data,random_crossover = False):
        self.data = data
        self.walkerArray = []
        self.walkerScore = []
        self.epochScore = []
        self.random_crossover = random_crossover
        
    def addWalker(self,initWalker=None):
        if initWalker == None:
            self.walkerArray.append(self.mutate(np.ones(self.data.shape))) #self.randomRFIMap())
        else:
            if np.random.rand()<0.2:
                self.walkerArray.append(self.mutate(initWalker))
            else:
                self.walkerArray.append(initWalker)
        
    def randomRFIMap(self):
        rfiMap = np.random.randint(0,2,size=np.shape(self.data)).astype(bool)
        return rfiMap
    
    def delTrans(self,walker_to_score):
        WA_ = sfft.fftshift(sfft.fft(self.data*self.walkerArray[walker_to_score],axis=1),axes=1)
        #_WA = np.fft.fftshift(np.fft.fft(WA_,axis=0),axes=0)
        ma_WA = self.mask()*WA_
        return np.sum(np.abs(ma_WA))
        
    def mask(self):
        hole = np.ones_like(self.data)
        sh = np.shape(hole)
        hole[:,sh[1]/2 - 20:sh[1]/2 + 20] = 0.
        #for i in range(-6,6):
        #    for j in range(-50,50):
        #        hole[sh[0]/2 + i,sh[1]/2 + j] = 0.
        return hole
        
    def score(self,walker_to_score):
        delfringeScore = self.delTrans(walker_to_score)
        self.walkerScore.append(delfringeScore)
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
            if np.random.rand()>0.00: # I turned off RFI in frequency mutations
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
        
    def resetWalkers(self):
        self.walkerArray = []
        self.walkerScore = []
    
    def findBestWalker(self):
        #if np.argmin(np.sum(np.sum(self.walkerArray,1),1)) == np.argmin(self.walkerScore):
        # Introduce random mating of top 2 scorers of population in frequency
        offspring = np.zeros_like(self.data)
        top1 = self.walkerArray[np.argmin(self.walkerScore)]
        secondBestIndx = np.argwhere(self.walkerScore==np.sort(self.walkerScore)[1])
        print len(secondBestIndx),secondBestIndx
        try:
            top2 = self.walkerArray[secondBestIndx[0][0]]
        except:
            top2 = self.walkerArray[secondBestIndx[0]]
        if self.random_crossover:
            matingChain = np.random.randint(0,self.data.shape[1],size=self.data.shape[1]/np.random.randint(1,100))
        else:
            matingChain = np.random.randint(0,self.data.shape[1],size=self.data.shape[1]/2)
        top1[:,matingChain] = top2[:,matingChain]
        self.initWalker = top1 #self.walkerArray[np.argmin(self.walkerScore)]
        self.epochScore.append(self.walkerScore[np.argmin(self.walkerScore)])
        #else:
        #    self.epochScore.append(self.walkerScore[np.argmin(self.walkerScore)])

        #print np.argmin(self.walkerScore),np.argmax(np.sum(np.sum(self.walkerArray,1),1))
        
        
    def runEpoch(self,num_of_walkers,epoch):
        for i in range(num_of_walkers):
            if epoch == 0:
                self.addWalker()
            else:
                self.addWalker(self.initWalker)
            self.score(i)
        self.findBestWalker()
        self.resetWalkers()
        
        
