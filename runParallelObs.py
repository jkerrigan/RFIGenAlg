import sys
from IPython.core.display import clear_output
import timeit
import pylab as pl
from RFIGenAlg import rfiGenAlg
import pyuvdata
import numpy as np
import scipy.fftpack as sfft
import aipy as a
from joblib import Parallel, delayed
import multiprocessing
#### Note: Run GenAlg on single baseline. Add in ability to take this as starting position for all baselines in the same observation, maybe even for future observations?
uv = pyuvdata.miriad.Miriad()
uv.read_miriad('zen.2457458.30612.xx.HH.uvcU')

#bsl = uv.baseline_array==uv.antnums_to_baseline(20,22)
#try:
#    uv.read_miriad('zen.2456242.30605.uvcRREcACOTUcA')
#except:
#    pass
evolvedWalker = np.load('evolvedWalker.npz.npy')

start_time = timeit.default_timer()
baselines = np.unique(uv.baseline_array)

def ParallelizeGenAl(uv,bl):
    blIndx = uv.baseline_array==bl
    data = uv.data_array[blIndx,0,:,0]
    x = rfiGenAlg(data,random_crossover=True,initWalker=evolvedWalker)
    pop_size = 30
    epochs = 50
    for i in range(epochs):
        x.runEpoch(pop_size,i)
    return np.logical_not(x.initWalker)
    #uv.flag_array[blIndx,0,:,0] = np.logical_not(x.initWalker)

num_cores = -1
flags = Parallel(n_jobs=num_cores,backend="threading")(delayed(ParallelizeGenAl)(uv,i) for i in baselines)
ct = 0
for i in baselines:
    print i
    blIndx = i==uv.baseline_array
    print np.shape(blIndx)
    uv.flag_array[:,0,:,0][blIndx] = flags[ct]
    ct+=1
print np.shape(flags)
uv.write_miriad('zen.2457458.30612.xx.HH.uvcUGA')    
#print 'Generation per sec. :',(1.*epochs)/np.round(timeit.default_timer() - start_time)
print np.round(timeit.default_timer() - start_time)
#### Save the best walker for observation, this can be used as the start point
#### for all other baselines, which should afford a speedup
#np.save('evolvedWalker.npz',x.initWalker)

win = a.dsp.gen_window(1024,window='blackman-harris')

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
