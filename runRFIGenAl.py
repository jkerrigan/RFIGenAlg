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

def fold(data,ch_fold,labels=False):
    # We want to fold over in frequency                                                                                                                
    # this will be done for both waterfalls and labels                                                                                                 
    # data should be in (times,freqs) format                                       
    ntimes,nfreqs = np.shape(data)
    dfreqs = int(nfreqs/ch_fold)
    if labels:
        data_fold = np.zeros((ntimes,nfreqs)).reshape(ch_fold,ntimes,dfreqs)
    else:
        data_fold = np.zeros((ntimes,nfreqs)).reshape(ch_fold,ntimes,dfreqs)
    for i in range(ch_fold):
        if labels:
            data_fold[i,:,:] = data[:,i*dfreqs:(i+1)*dfreqs]
        else:
            #hold = np.nan_to_num(np.log10(np.abs(data[:,i*dfreqs:(i+1)*dfreqs]+np.random.rand(ntimes,dfreqs)))).real
            data_fold[i,:,:] = data[:,i*dfreqs:(i+1)*dfreqs]#(hold - np.nanmean(hold))/np.nanmax(np.abs(hold)) #theres a better way to do this                                     
    return data_fold.real

def unfold(data_fold,nchans):
    ch_fold,ntimes,dfreqs = np.shape(data_fold)
    data = np.zeros_like(data_fold).reshape(60,1024)
    for i in range(ch_fold):
        data[:,i*dfreqs:(i+1)*dfreqs] = data_fold[i,:,:]
    return data

uv = pyuvdata.miriad.Miriad()
uv.read_miriad('/users/jkerriga/data/jkerriga/IDR2_1/zen.2458107.33430.xx.HH.uvOC',run_check=False)
## Average data ##
#for ap in uv.get_antpairs():
#    if ap[0] == ap[1]:
#        continue
#    if ap == uv.get_antpairs()[1]:
#        data = np.copy(uv.get_data(ap))
#    else:
#        data += np.copy(uv.get_data(ap))
data = uv.get_data(2,12)
data_fold = fold(data,16)
sh = np.shape(data_fold)
print 'Shape of folded data: ',sh
#print 'Starting generation is an evolved generation.'
#evolvedWalker = np.load('evolvedWalker.npz.npy')
pad = 0
#x = rfiGenAlg(data,random_crossover=True,initWalker=evolvedWalker)

#x = rfiGenAlg(data_fold,random_crossover=True,initWalker=None,pad=pad,live=True)
GAflags = np.zeros_like(data_fold).astype(int)
pop_size = 1000
epochs = 30
e_ct = 0
start_time = timeit.default_timer()
for df in range(16):
    x = rfiGenAlg(data_fold[df,:,:],random_crossover=True,initWalker=None,pad=pad,live=True)
    for i in range(epochs):
        clear_output()
        print 'Generation: ',i
        print 'Time: ',np.round(timeit.default_timer() - start_time), ' s'
        sys.stdout.flush()
        x.runEpoch(pop_size,i)
        print x.epochScore
        if i > 1:
            if x.epochScore[i] >= x.epochScore[i-1]:
                break
        print np.shape(x.initWalker)
    GAflags[df,:,:] = x.initWalker[pad:pad+sh[1],:]
    if df != 15:
        del(x)

#    if i > 2000:
#        if np.std(x.epochScore[-600:]) < np.mean(x.epochScore[-600:])-x.epochScore[i]:
#            print i
#            break
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
GAflags = unfold(GAflags,1024)

#### Save the best walker for observation, this can be used as the start point
#### for all other baselines, which should afford a speedup
#np.save('evolvedWalker.npz',x.initWalker)

#win = a.dsp.gen_window(1024,window='blackman-harris')
def delTrans(data,flags):
    win = a.dsp.gen_window(1024,window='blackman-harris')
    DATA = sfft.fftshift(sfft.fft(win*data*flags,axis=1),axes=1)
    DATA_ = sfft.fftshift(sfft.fft(DATA,axis=0),axes=0)
    return DATA_
win = a.dsp.gen_window(1024,window='blackman-harris')
run_time = np.round(timeit.default_timer() - start_time) 
pl.subplot(311)
pl.imshow(np.log10(np.abs(data[pad:pad+sh[0]])),aspect='auto',interpolation='none',cmap='jet',vmax=np.max(np.log10(np.abs(data[pad:pad+sh[0]]))),vmin=-3)
pl.subplot(312)
#pl.imshow(np.log10(np.abs(win*data*x.initWalker[pad:pad+sh[0],:])),aspect='auto',interpolation='none',cmap='jet')
pl.imshow(np.log10(np.abs(win*data*GAflags)),aspect='auto',interpolation='none',cmap='jet',vmax=np.max(np.log10(np.abs(win*data*GAflags))),vmin=-3)
#pl.subplot(413)
#pl.imshow(np.log10(np.abs(delTrans(data,x.initWalker[pad:pad+sh[0],:])*x.mask()[pad:pad+sh[0],:])),aspect='auto',interpolation='none',cmap='jet')
#pl.imshow(np.log10(np.abs(delTrans(data,GAflags)*x.mask()[pad:pad+sh[0],:])),aspect='auto',interpolation='none',cmap='jet')
#x.plotRFIMap(live=False)
pl.subplot(313)
#pl.imshow(np.log10(np.abs(sfft.fftshift(sfft.fft(win*data*x.initWalker[pad:pad+sh[0],:],axis=1),axes=1))),aspect='auto',interpolation='none',cmap='jet')
pl.imshow(np.log10(np.abs(sfft.fftshift(sfft.fft(win*data*GAflags,axis=1),axes=1))),aspect='auto',interpolation='none',cmap='jet')
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
