import sys
from IPython.core.display import clear_output
import timeit
import pylab as pl
from RFIGenAlg import rfiGenAlg
import pyuvdata
import numpy as np
#### Note: Run GenAlg on single baseline. Add in ability to take this as starting position for all baselines in the same observation, maybe even for future observations?
uv = pyuvdata.miriad.Miriad()
uv.read_miriad('zen.2457458.30612.xx.HH.uvcUA')
data = uv.data_array[:,0,:,0]
print data.shape
x = rfiGenAlg(data)

pop_size = 20
epochs = 30000
start_time = timeit.default_timer()
for i in range(epochs):
    clear_output()
    print 'Generation: ',i
    print 'Time: ',np.round(timeit.default_timer() - start_time), ' s'
    sys.stdout.flush()
    x.runEpoch(pop_size,i)
print 'Generation per sec. :',(1.*epochs)/np.round(timeit.default_timer() - start_time)

run_time = np.round(timeit.default_timer() - start_time) 
pl.subplot(211)
pl.imshow(np.log10(np.abs(data)),aspect='auto',interpolation='none')
pl.subplot(212)
x.plotRFIMap(live=False)
pl.savefig('RFIEvolved_'+str(run_time)+'.png')
pl.close()

pl.plot(np.log10(x.epochScore))
pl.xlabel('Generation')
pl.ylabel('log10(Fitness Score)')
pl.savefig('RFIGenEvolution_'+str(run_time)+'.png')
pl.close()
