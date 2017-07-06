import sys
from IPython.core.display import clear_output
import timeit
import pylab as pl
from RFIGenAlg import rfiGenAlg
import pyuvdata
import numpy as np
import scipy.fftpack as sfft
#### Note: Run GenAlg on single baseline. Add in ability to take this as starting position for all baselines in the same observation, maybe even for future observations?
uv = pyuvdata.miriad.Miriad()
uv.read_miriad('/Users/josh/Desktop/HERA/data/zen.2457458.17389.xx.HH.uvcUA')
#bsl = uv.baseline_array==uv.antnums_to_baseline(20,22)
#try:
#    uv.read_miriad('zen.2456242.30605.uvcRREcACOTUcA')
#except:
#    pass
#bsl = uv.baseline_array==uv.antnums_to_baseline(41,49)
data = uv.data_array[:,0,:,0]
print data.shape
x = rfiGenAlg(data,random_crossover=True)

pop_size = 15
epochs = 1000
e_ct = 0
start_time = timeit.default_timer()
for i in range(epochs):
    clear_output()
    print 'Generation: ',i
    print 'Time: ',np.round(timeit.default_timer() - start_time), ' s'
    sys.stdout.flush()
    x.runEpoch(pop_size,i)
    if np.log10(x.epochScore[i]) < 4.5:
        break
print 'Generation per sec. :',(1.*epochs)/np.round(timeit.default_timer() - start_time)

run_time = np.round(timeit.default_timer() - start_time) 
pl.subplot(411)
pl.imshow(np.log10(np.abs(data)),aspect='auto',interpolation='none',cmap='jet')
pl.subplot(412)
pl.imshow(np.log10(np.abs(data*x.initWalker)),aspect='auto',interpolation='none',cmap='jet')
pl.subplot(413)
x.plotRFIMap(live=False)
pl.subplot(414)
pl.imshow(np.log10(np.abs(sfft.fftshift(sfft.fft(data*x.initWalker)))),aspect='auto',interpolation='none',cmap='jet')

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
