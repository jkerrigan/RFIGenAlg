import sys
from IPython.core.display import clear_output
import timeit
import pylab as pl

x = rfiGenAlg(data)
pop_size = 400
epochs = 3000
start_time = timeit.default_timer()
for i in range(epochs):
#    clear_output()
    print 'Generation: ',i
    print 'Time: ',np.round(timeit.default_timer() - start_time), ' s'
#    sys.stdout.flush()
    x.runEpoch(pop_size,i)
print 'Generation per sec. :',(1.*epochs)/np.round(timeit.default_timer() - start_time)

run_time = np.round(timeit.default_timer() - start_time) 
x.plotRFIMap()
pl.savefig('RFIEvolved_'+str(run_time)+'.png')
pl.close()

pl.plot(np.log10(x.epochScore))
pl.xlabel('Generation')
pl.ylabel('log10(Fitness Score)')
pl.savefig('RFIGenEvolution_'+str(run_time)+'.png')
pl.close()
