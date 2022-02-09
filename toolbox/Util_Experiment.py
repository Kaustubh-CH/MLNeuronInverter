import numpy as np
import hashlib
import string,random # for id-gen

#...!...!..................
def rebin_data1D(X,nReb,clip=False):
        tBin=X.shape[0]
        #print('X1D',X.shape,nReb)
        outBin=tBin//nReb
        if clip: # clip the reminder of array
            X=X[:outBin*nReb]
        a=X.reshape(outBin,nReb)
        b=np.sum(a,axis=1)/nReb
        #print('reb X1',a.shape,b.shape,b.dtype)
        return b

#...!...!..................
def rebin_data2D(X,nReb,clip=False):
        nS,tBin=X.shape
        #print('X2D',X.shape,'nReb=',nReb)
        outBin=tBin//nReb
        if clip: # clip the reminder of array
            X=X[:,:outBin*nReb]
        a=X.reshape(nS,outBin,nReb)
        b=np.sum(a,axis=2)/nReb
        #print('reb X2',a.shape,b.shape,b.dtype)
        return b


#...!...!..................
def md5hash(text,size=6):
    hao = hashlib.md5(text.encode())
    hastr=hao.hexdigest()
    return hastr[-size:]  # use just last 6 characters from 32-long hash

#...!...!.................. # returns random sequence of letters and numbers
def id_generator(size=6, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

#...!...!..................

#...!...!..................


#............................
#............................
#............................
class SpikeFinder(object):
    def __init__(self, timeAx,verb=1):
        # configuraction of spike-finder
        confD={'min_raise_speed_V/s': 30, 'min_peak_ampl_mV': -10}

        self.conf=confD
        self.timeAx=timeAx
        print('JanSpikeFinder, conf:',confD,'timeAx=',timeAx.shape)
        # auxialiry params
        self.par_di=2  # (time bins)  distance from spike to sliding edge
        #Xself.par_break_speed_V2sec=1.  # (V/sec)  breaking speed for terminating spike
        # this will have 13 spikes with brea speed=30: 210611_6_NI_023_96
        self.tstep=(timeAx[2] -timeAx[1])
        self.break_amp_step=confD['min_raise_speed_V/s']*self.tstep
        self.verb=verb

#...!...!..................
    def make(self,wave1):  # returnes time-sorted list of spikes
        if self.verb>1: print('JSS:make  wave shape:',wave1.shape)
        assert wave1.shape==self.timeAx.shape
        assert not np.isnan(wave1).any()
        self.wave=np.copy(wave1) # analysis is destructive
        avrPreMemBias=np.mean(wave1[:50])
        
        #self.spikes=[]
        self.spikesD={}
        if avrPreMemBias > -30:
            print('JSPM corruped waveform, skip it')
            print('wave1',avrPreMemBias,wave1[:50])
            return 0
        while True:
            if self.find_spike()<0: break
            #break #tmp

        #... sort spikes by tPeak
        tX=sorted(self.spikesD)
        print('tX',type(tX),tX)
        self.spikes=[ [x]+self.spikesD[x] for x in tX]
        return len(self.spikes)

#...!...!..................
    def find_spike(self):
        idx=np.argmax(self.wave)
        ampPeak=self.wave[idx]
        tPeak=self.timeAx[idx]
        accept=ampPeak>self.conf['min_peak_ampl_mV']
        if self.verb>1: print('\n  FP:new max t=%.2f ms  y=%.1f mV  accept=%r idx=%d'%(tPeak,ampPeak,accept,idx))
        #print('www',self.wave)
        if not accept : return -1

        isSpike,idxL=self.slideLeft(idx)
        # if isSpike==False still find it to erase it - just do not save it
        yL=self.wave[idxL]
        if self.verb>1: print('idxL=%d  t=%.2f ns, y=%.1f mV isSpike=%r'%(idxL,self.timeAx[idxL],yL,isSpike))
        

        idxR=self.slideRight(idx,yL)
        yR=self.wave[idxR]
        if self.verb>1: print('idxR=%d  t=%.2f ns, y=%.1f mV '%(idxR,self.timeAx[idxR],yR))
        

        twidthBase=(idxR-idxL)*self.tstep
        ampBase=(yL+yR)/2.
        tBase=self.timeAx[idxL]

        if self.verb>1: print('base: twidth=%.1f ms, base amp=%.1f mV'%(twidthBase,ampBase))

        idx_span=[idxL,idxR]
        ref_amp=(ampPeak+ampBase)/2.
        isFwhm,ixdL,idxR=self.getFWHM(idx,ref_amp,idx_span)
        yR=self.wave[idxR]
        #print('fwhm: idxR=%d  t=%.2f ns, y=%.1f mV '%(idxR,self.timeAx[idxR],yR))
        
        yL=self.wave[idxL]
        twidth=(idxR-idxL)*self.tstep
        if self.verb>1: print(' half-peak twidth=%.2f ms amp=%.1f spikeId=%d'%(twidth,(yR+yL)/2.,len(self.spikes)))
        #  isFwhm==False for  rare cases spike is at the edge of time-axis
        
        if isSpike and isFwhm:            
            #self.spikes.append([tPeak,ampPeak,twidth,tBase,ampBase,twidthBase])
            self.spikesD[tPeak]=[ampPeak,twidth,tBase,ampBase,twidthBase]
                
        # clear data over base range
        for i in range(idx_span[0],idx_span[1]+1):
            self.wave[i]=ampBase

        return 0

#...!...!..................
    def slideLeft(self,idx):
        isSpike=False
        if self.verb>2: print('dump[-10:10]',self.wave[idx-10:idx+10])
        for i in range(idx-self.par_di,0,-1):
            delAmp= self.wave[i+1] -self.wave[i]
            if self.verb>2: print('L i=%d a=%.1f d=%.1f is=%r'%(i,self.wave[i],delAmp/self.tstep,isSpike))
            if delAmp> self.break_amp_step: isSpike=True # dvdt is large enough
            if delAmp <0 : break # waveform raises again - end of spike cadidate
            if not isSpike: continue
            # find spike terminating condition
            if delAmp < self.break_amp_step:  break # nominal spike baseline
                        
        return isSpike,i


#...!...!..................
    def slideRight(self,idx,minAmp):
        i=idx # default for rare cases spike is at the edge of time-axis
        for i in range(idx+self.par_di,self.wave.shape[0]):
            delAmp= self.wave[i-1] -self.wave[i]
            if self.verb>2:  print('R %d %.1f %.1f '%(i,self.wave[i],delAmp/self.tstep))
            
            if delAmp <0: break  # next spike starts?
            if self.wave[i]< minAmp:   break # good one spike

        return i

#...!...!..................
    def getFWHM(self,idx,amp,ispan):
        iL=iR=idx
        for i in range(idx,ispan[0],-1):
            if self.wave[i] < amp: break
            iL=i

        for i in range(idx,ispan[1]):
            if self.wave[i] < amp: break
            iR=i

        ok= ispan[0] <iL  and iR <ispan[1]
        return ok,iL,iR
    
