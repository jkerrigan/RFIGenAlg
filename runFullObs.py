import sys
from IPython.core.display import clear_output
import timeit
import pylab as pl
from RFIGenAlg import rfiGenAlg
import pyuvdata
import numpy as np
import scipy.fftpack as sfft
import aipy as a
import os,sys,optparse

o = optparse.OptionParser()
o.set_usage('pspec_prep.py [options] *.uv')
opts, args = o.parse_args(sys.argv[1:])
#### Note: Run GenAlg on multiple observations. Add in ability to take this as starting position for all baselines in the same observation, maybe even for future observations?
#occStat = np.zeros(1024)
pad = 10
for obs in args:
    print obs
    uv = pyuvdata.miriad.Miriad()
    uv.read_miriad(obs)

    start_time = timeit.default_timer()
    baselines = np.unique(uv.baseline_array)
#evolvedWalker = np.load('evolvedWalker.npz.npy')
    for bl in baselines[1:]:
        blIndx = uv.baseline_array==bl
        data = uv.data_array[blIndx,0,:,0]
        sh = np.shape(data)
#x = rfiGenAlg(data,random_crossover=True,initWalker=evolvedWalker)
        if bl == baselines[1]:
            x = rfiGenAlg(data,random_crossover=True,pad=10,initWalker=None)
            pop_size = 50
            epochs = 1000
            for i in range(epochs):
                x.runEpoch(pop_size,i)
                #if i > 1000:
                #    if np.std(x.epochScore[-500:]) < np.mean(x.epochScore[-500:])-x.epochScore[i]:
                #        print i
                #        break
        #else:
            #pop_size = 3
            #epochs = 3
            #x = rfiGenAlg(data,random_crossover=True,initWalker=carryOverEvol)
        #for i in range(epochs):
        #    x.runEpoch(pop_size,i)
        flags = x.initWalker[pad:pad+sh[0],:]
        if bl == baselines[1] and obs == args[0]:
            occStat = np.sum(flags,0)/(1.*sh[0]) 
        elif obs != args[0] and bl == baselines[1]:
            occStat = np.mean((occStat,np.sum(flags,0)/(1.*sh[0])),0)
        uv.flag_array[blIndx,0,:,0] = np.logical_not(flags)
        #carryOverEvol = np.copy(x.initWalker)
        #carryOverEpochs = np.copy(x.epochScore)
        #del(x)
    #del(carryOverEvol)
    uv.write_miriad(obs+'GA')
    print 'Time: ',np.round(timeit.default_timer() - start_time), ' s'
    del(uv)
#print 'Generation per sec. :',(1.*epochs)/np.round(timeit.default_timer() - start_time)
print np.round(timeit.default_timer() - start_time)
#### Save the best walker for observation, this can be used as the start point
#### for all other baselines, which should afford a speedup
#np.save('evolvedWalker.npz',x.initWalker)

win = a.dsp.gen_window(1024,window='blackman-harris')
freqs = np.linspace(100,200,1024)
pl.plot(freqs,occStat)
pl.show()
run_time = np.round(timeit.default_timer() - start_time) 
pl.subplot(411)
pl.imshow(np.log10(np.abs(data)),aspect='auto',interpolation='none',cmap='jet')
pl.subplot(412)
pl.imshow(np.log10(np.abs(win*data*carryOverEvol)),aspect='auto',interpolation='none',cmap='jet')
pl.subplot(413)
#x.plotRFIMap(live=False)
pl.subplot(414)
pl.imshow(np.log10(np.abs(sfft.fftshift(sfft.fft(data*carryOverEvol,axis=1),axes=1))),aspect='auto',interpolation='none',cmap='jet')
pl.savefig('RFIEvolved_'+str(run_time)+'.png')
pl.close()

pl.subplot(111)
pl.plot(np.log10(carryOverEpochs))
pl.xlabel('Generation')
pl.ylabel('log10(Fitness Score)')
#pl.subplot(312)
#pl.plot(np.log10(x.mutRateArray))
#pl.xlabel('Generation Family')
#pl.ylabel('Mutation Rate')
#pl.subplot(313)
#pl.plot(np.log10(x.growthVolArray))
#pl.xlabel('Generation Family')
#pl.ylabel('Mutation Growth Volatility')

pl.savefig('RFIGenEvolution_'+str(run_time)+'.png')
pl.close()
