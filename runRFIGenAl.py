import sys
from IPython.core.display import clear_output
import timeit
import pylab as pl
from RFIGenAlg import rfiGenAlg
import pyuvdata
import numpy as np
import scipy.fftpack as sfft
import aipy as a
#### Note: Run GenAlg on single baseline. Add in ability to take this as starting position for all baselines in the same observation, maybe even for future observations?
uv = pyuvdata.miriad.Miriad()
uv.read_miriad('/Users/josh/Desktop/HERA/data/zen.2457458.17389.xx.HH.uvcUA')
#bsl = uv.baseline_array==uv.antnums_to_baseline(20,22)
#try:
#    uv.read_miriad('zen.2456242.30605.uvcRREcACOTUcA')
#except:
#    pass
bsl = uv.baseline_array==uv.antnums_to_baseline(9,10)
data = uv.data_array[bsl,0,:,0]

print 'Starting generation is an evolved generation.'
#evolvedWalker = np.load('evolvedWalker.npz.npy')

#x = rfiGenAlg(data,random_crossover=True,initWalker=evolvedWalker)
x = rfiGenAlg(data,random_crossover=True,initWalker=None)
pop_size = 30
epochs = 4000
e_ct = 0
start_time = timeit.default_timer()
for i in range(epochs):
    clear_output()
    print 'Generation: ',i
    print 'Time: ',np.round(timeit.default_timer() - start_time), ' s'
    sys.stdout.flush()
    x.runEpoch(pop_size,i)
#    try:
#        if 1 - x.epochScore[i]/np.average(x.epochScore[i-10:i]) < .001:
#            if pop_size < 500:
#                pop_size*=2
#        else:
#            if pop_size > 10:
#                pop_size/=2
#    except:
#        pass
#    if i > 50:
#        if 1 - x.epochScore[i]/np.average(x.epochScore[i-100:i]) < .001:
#            break

    print pop_size
print 'Generation per sec. :',(1.*epochs)/np.round(timeit.default_timer() - start_time)

#### Save the best walker for observation, this can be used as the start point
#### for all other baselines, which should afford a speedup
np.save('evolvedWalker.npz',x.initWalker)

#win = a.dsp.gen_window(1024,window='blackman-harris')
def delTrans(data,flags):
    win = a.dsp.gen_window(1024,window='blackman-harris')
    DATA = sfft.fftshift(sfft.fft(win*data*flags,axis=1),axes=1)
    DATA_ = sfft.fftshift(sfft.fft(DATA,axis=0),axes=0)
    return DATA_
win = a.dsp.gen_window(1024,window='blackman-harris')
run_time = np.round(timeit.default_timer() - start_time) 
pl.subplot(411)
pl.imshow(np.log10(np.abs(data)),aspect='auto',interpolation='none',cmap='jet')
pl.subplot(412)
pl.imshow(np.log10(np.abs(win*data*x.initWalker)),aspect='auto',interpolation='none',cmap='jet')
pl.subplot(413)
pl.imshow(np.log10(np.abs(delTrans(data,x.initWalker))),aspect='auto',interpolation='none',cmap='jet')
#x.plotRFIMap(live=False)
pl.subplot(414)
pl.imshow(np.log10(np.abs(sfft.fftshift(sfft.fft(win*data*x.initWalker,axis=1),axes=1))),aspect='auto',interpolation='none',cmap='jet')
pl.savefig('RFIEvolved_'+str(run_time)+'.png')
pl.close()

pl.subplot(311)
pl.plot(np.log10(x.epochScore))
pl.xlabel('Generation')
pl.ylabel('log10(Fitness Score)')
pl.subplot(312)
pl.plot(np.log10(x.mutRateArray))
pl.xlabel('Generation Family')
pl.ylabel('Mutation Rate')
pl.subplot(313)
pl.plot(np.log10(x.growthVolArray))
pl.xlabel('Generation Family')
pl.ylabel('Mutation Growth Volatility')

pl.savefig('RFIGenEvolution_'+str(run_time)+'.png')
pl.close()
