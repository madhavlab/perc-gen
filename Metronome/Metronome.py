print
DEBUG = True 


import pyaudio
import json
import time
import sys, pdb, os 
import numpy as np 
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import threading
from pynput import keyboard
from scipy.io.wavfile import write
import serial.tools.list_ports
import random
import pandas as pd     
from queue import Queue         
from re import X
serialInst = serial.Serial()
global store
global store2
store = np.empty((0), float)
store2 = np.empty((0), float)
store3 = np.empty((0), float)
store4 = np.empty((0), float)


def plotWav(y, RATE):
    x = np.arange(len(y))/RATE 
    plt.plot(x, y)
    return

def writeWav(y, RATE, outfile='out.wav'):
    sf.write(outfile, y, RATE, 'PCM_24')
    return 

class dataIO:
    'Conventions for strokes'
    def __init__(self,name_cc, strokesWavCsv='csv/strokeWavs.csv', 
            beatCyclesCsv='csv/beatCycles.csv', fillersCsv='csv/fillers.csv', callCycleCsv = 'csv/callcycles.csv'):
        self.BLANK = '-'
        self.stroke2wavdelay = self._readStrokeWavsCsv(strokesWavCsv)
        self.wavDIR = 'wav/'
        self.RATE = 16000   # fixed
        self.name2callcycle, self.name2callcycleprob = self._readCallCycleCsv(callCycleCsv, name_cc)
        self.name2beatCycle = self._readBeatCyclesCsv(beatCyclesCsv)
        self.name2filler = self._readFillersCsv(fillersCsv)
        # pdb.set_trace()
    def getStrokeWav(self, stroke):
        wavfile, delay_sec = self.stroke2wavdelay[stroke]
        wavdata, sr = librosa.load(os.path.join(self.wavDIR, wavfile), sr=self.RATE, mono=False)
        if wavdata.ndim > 1:
            wavdata = wavdata[0] # one channel
        delay_samples = int(delay_sec*self.RATE)
        return wavdata, delay_samples
    def _readStrokeWavsCsv(self, strokesWavCsv):
        'stroke,wav,delay_seconds'  
        # strokes = set([stroke for beat in self.beatCycle for stroke in beat])
        stroke2wavdelay = {}
        stroke2wavdelay[self.BLANK] = [None, 0]
        with open(strokesWavCsv,'r') as fid: # read csv
            for line in fid:
                # stroke2line[line.split(',')[0]] = line
                tokens = line.strip().split(' ')
                # pdb.set_trace()
                wavfile = tokens[1]
                delay_sec = float(tokens[2]) 
                stroke2wavdelay[tokens[0]] = [wavfile, delay_sec]
        return stroke2wavdelay
    def _readBeatCyclesCsv(self, beatCyclesCsv):
        'name   BeatCycle'
        name2beatCycle = {}
        with open(beatCyclesCsv,'r') as fid: # read csv
            for line in fid:
                tokens = line.strip().split(' ')
                name = tokens[0]
                beatcycle = json.loads(tokens[1])
                name2beatCycle[name] = beatcycle
        return name2beatCycle
    def _readCallCycleCsv(self, callcyclecsv, name_cc):
        # name2callcycle = {}
        name2callcycle = []
        name2callcycleprob = []
        with open(callcyclecsv,'r') as fid: # read csv
            for line in fid:
                tokens = line.strip().split(' ')    
                name = tokens[0]
                if (name_cc == name):
                    # print(name)
                    for i in range(1,len(tokens), 2):
                        name2callcycle.append(tokens[i])
                        name2callcycleprob.append(float(tokens[i+1]))
        return name2callcycle, name2callcycleprob

    def _readFillersCsv(self, fillersCsv):
        'name   filler  anchor'
        name2filler = {}
        with open(fillersCsv,'r') as fid: # read csv
            for line in fid:
                tokens = line.strip().split(' ')
                name = tokens[0]
                # pdb.set_trace()
                filler = json.loads(tokens[1])
                anchor = int(tokens[2]) # the last beat of filler is placed here in the next call cycle
                name2filler[name] = filler, anchor
        return name2filler
    
class SamplesPlate:
    def __init__(self, beatsplate, BPM, name_cc):
        self.dataio = dataIO(name_cc)
        self.plate = np.zeros(self.dataio.RATE * 5  )  # np.array (N,)
        self.pointer_currentstroke2refill = 512    # start refill here # initial delay
        self.BPM = BPM       # can change dynamically
        self.stroke2wav = {}  # get wav from here
        self.beatsplate = beatsplate
        self.ON = True
        self.framelen = 100     # to activate refill in keepFilled
        self.beatlen = int(self.dataio.RATE * 60 / self.BPM)
        self.refill(3) # This cant be greater that beats in the beatcycle

    def fetch(self, framelen): # get the buffer
        self.framelen = framelen 
        # print('fetch - framelen: - ', self.framelen )
        retval = self.plate[:framelen]*1    
        # print('fetch - retva  l len: - ', len(retval))
        self.plate[:-framelen] = self.plate[framelen:]  # move the queue 
        self.plate[-framelen:] *= 0
        self.pointer_currentstroke2refill -= framelen  # refill pointer also moves
        # print('fetch - pointer_currentstroke2refill: - ',self.pointer_currentstroke2refill)
        self.keepFilled()
        return retval

    def refill(self, Nbeats):
        'add more strokes to the plate'
        if DEBUG: 
            print('\nRefill with BPM:', self.BPM)
            print('Beatlen', self.beatlen)  ######################
            current_time = time.perf_counter() - mg_start_time
            print('<-----------------Current time: ',current_time)
            beat_start_time = current_time

        RATE = self.dataio.RATE 
        beats = self.beatsplate.fetch(Nbeats) # [['tin'],['tin'],['-'],]
        if DEBUG: print('beats:', beats) ########################################
        for beat in beats:
            Nstrokes = len(beat)    # a beat can have multiple strokes, e.g. ['te','re']
            strokelen = int(self.beatlen/Nstrokes) # samples
            for stroke in beat:
                if stroke==self.dataio.BLANK:
                    wav, delay_samples = [np.zeros(strokelen), 0]
                else:
                    wav, delay_samples = self.dataio.getStrokeWav(stroke)
                # ADD
                # n_start = self.pointer_currentstroke2refill - delay_samples
                # n_end = n_start + len(wav)
                # self.plate[n_start:n_end] += wav 
                # self.pointer_currentstroke2refill += strokelen 
                p = self.pointer_currentstroke2refill - delay_samples
                if p<0: # trim non-causal part
                    wav = wav[-p:] 
                    p = 0
                try:
                    if DEBUG:
                        print('p', p, 'len(wav)', len(wav), 'wav.shape', wav.shape) ################################
                    self.plate[p:p+len(wav)] += wav
                except ValueError:
                    self.extendPlate(p+len(wav))
                    # print(len(self.plate),p+len(wav),)
                    self.plate[p:p+len(wav)] += wav
                self.pointer_currentstroke2refill += strokelen # move pointer 
        return 

    def extendPlate(self, length):
        'extend the total length of plate to hold longer stroke wavs' 
        assert length > len(self.plate), (length, len(self.plate))
        self.plate = np.concatenate((self.plate, np.zeros(length-len(self.plate))))

    def keepFilled(self):
        # self.beatlen = int(self.dataio.RATE * 60 / MetronomeGenerator.currentBPM)
        Nbeats = 1        # CRUCIAL - the latency in changing BPM
        while self.ON:
            if self.pointer_currentstroke2refill < Nbeats * self.beatlen:   
                self.refill(Nbeats) # This cant be greater than beats in the beatcycle
            else:
                break
                # time.sleep((self.pointer_currentstroke2refill/self.beatlen - Nbeats)/self.BPM * 60)

    def setBPM(self, BPM):
        self.BPM = BPM
        self.beatlen = int(self.dataio.RATE * 60 / self.BPM)
        return 

    def turnOFF(self):
        self.ON = False
        return 

class BeatsPlate():
    '''Supplies beats: E.g. [['tin'],['tin'],['-'],]
    '''         
    def __init__(self, beatcyclesplate):
        self.plate = []
        self.beatcyclesplate = beatcyclesplate
        self.Nbeats = 1     # to activate refill in keepFilled
        self.ON = True
        self.refill()
        return 

    def fetch(self, Nbeats):
        self.Nbeats = Nbeats 
        retval = self.plate[:Nbeats]
        del self.plate[:Nbeats] # move the queue 
        self.keepFilled()
        return retval 

    def refill(self):
        # if DEBUG: print('Beatsplate - refill') ##################################
        beats = self.beatcyclesplate.fetch()
        self.plate += beats
        return 

    def keepFilled(self):
        if self.ON:
            if len(self.plate)<3*self.Nbeats: 
                self.refill()
        return

    def setBeatCyclesPlate(self, beatcyclesplate):
        self.beatcyclesplate = beatcyclesplate
        # print("beatsplate - setBeatCyclesPlate:-", self.beatcyclesplate)
        return

class BeatCyclesPlate():
    '''
    - Supplies beatCycles: E.g. [['tin'],['tin'],['dha'],['-']]
    - Plate carries the entire callCycle; it is a circular queue
    - to update the taal, create a new BeatCyclesPlate and attach it to original BeatsPlate
    '''
    def __init__(self, name_cc, name_filler, NbeatCyclesPerCall):
        self.plate = []
        self.name_cc = name_cc
        self.name_filler = name_filler
        self.dataio = dataIO(self.name_cc)
        self.NbeatCyclesPerCall = NbeatCyclesPerCall
        # pdb.set_trace()
        self._NbeatsperBC = 2 # changed this, should be less than beats is any sequence
        self.ON = True
        # self.nbeats = nbeats
        self.refill()

    def fetch(self, ):
        'fetches one beat cycle'
        global nbeats
        # global putfiller

        retval = self.plate[:self._NbeatsperBC]
        del self.plate[:self._NbeatsperBC] # move the queue 
        # self.nbeats +=1
        # if (self.nbeats == putfiller-1 ):
            # self.NbeatCyclesPerCall = self.nbeats
        self.keepFilled()
        return retval   

    def refill(self):
        'refill one call cycle'
        beats = []
        if self.name_filler is None:

            callcyclelist = self.dataio.name2callcycle
            probabilities = self.dataio.name2callcycleprob
            key = random.choices(callcyclelist, weights = probabilities)
            beats = self.dataio.name2beatCycle[key[0]]
            callcycle = beats
        else: # add filler at the end of call cycle
            callcyclelist = self.dataio.name2callcycle
            probabilities = self.dataio.name2callcycleprob
            key = random.choices(callcyclelist, weights = probabilities)
            beats = self.dataio.name2beatCycle[key[0]]
            callcycle = beats * self.NbeatCyclesPerCall

            randkey = random.choices(list(self.dataio.name2filler))
            filler, anchor = self.dataio.name2filler[randkey[0]]
            # filler = filler.astype('object')
            # callcycle = self.extendPlate(len(filler), callcycle)
            if anchor==0:
                # callcycle += filler
                callcycle[-len(filler):] = filler[:-1]
            else:
                assert False, ['Code not written for anchor', anchor]
        self.plate += callcycle
        return

    def keepFilled(self):
        if self.ON:
            if len(self.plate)<self._NbeatsperBC: 
                self.refill()
        return

    def extendPlate(self, length, callcycle):
        'extend the total length of plate to hold longer stroke wavs' 
        callcycle = np.concatenate((callcycle, np.zeros((length,1), dtype = object)))
        return callcycle

    def setnamefiller(self,name_filler):
        self.name_filler = name_filler


obs =0                  
obsv =0 
bpm = 0 
indexing = 0.0  
class MetronomeGenerator(): 
    def __init__(self):   
                

        global mg_start_time
        mg_start_time = time.perf_counter()     
    
        self.start_time = time.perf_counter()
        self.last_key_press_time = self.start_time
        self.current_time1 = None
        self.elapsed_time = None

        
        self.i=0            
        self.currentBPM = 60                                                                                                                                                                                                                   
        self.paused = False 
        self.bypass = 0     
        self.hardware(self.bypass)
        # q=Queue()         
        self.y_record = np.empty((10000), float)                   
        self.x_record = np.empty((0), int)              # Sensor input

        # self.putfiller = 0            
        self.x=0    
        self.leftptr=0  
        self.currempty=0                            
        self.prevreading=0      
        self.sum=0  
        self.esum =0
        self.flag=0         
        name_cc = 'Kartaal' 
        # name_filler = 'MridangaFiller'    
        self.name_filler = None 
        # self.name_filler = 'randomised'   
        NbeatCyclesPerCall = 2      

        self.beatcyclesplate = BeatCyclesPlate(name_cc, self.name_filler, NbeatCyclesPerCall)
        x= self.beatcyclesplate._NbeatsperBC
        self.beatsplate = BeatsPlate(self.beatcyclesplate)

        self.samplesplate = SamplesPlate(self.beatsplate, self.currentBPM, name_cc)
        self.RATE = self.samplesplate.dataio.RATE
 
        self.thread_stream = threading.Thread(target=self.streaming, name='thread_stream')
        self.sensor_input = threading.Thread(target=self.sensorinput, name = 'sensor_input')
        self.thread_stream.start()  
        self.sensor_input.start()


    def streaming(self,):

            CHUNK = 512  
            p = pyaudio.PyAudio()
            self.stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=int(self.RATE),
                        input=True,
                        output=True,
                        frames_per_buffer=CHUNK,
                        stream_callback=self.callback)  

            self.stream.start_stream()  
            while self.stream.is_active() or self.paused==True:
                with keyboard.Listener(on_press=self.on_press) as listener:
                    listener.join()
                time.sleep(0.1)

            self.stream.stop_stream()
            self.stream.close()

            p.terminate()


    def hardware(self, bypass):
        
            global serialInst

            if (bypass==1):
                return
            else:
                serialInst.baudrate = 9600
                serialInst.port = "COM3" # Automate the code
                serialInst.open()

    def start(self): # !!!!!
            self.samplesplate.ON = True
            self.thread_fillsamplesplate.start() # keeps refilling 
            self.thread_stream.start()
            return  

    def stop(self): # !!!!!
            self.samplesplate.ON = False
            self.thread_fillsamplesplate.join()
            self.thread_stream.join()
            self.stream.stop_stream()
            self.stream.close()
            # p.terminate()
            return

    def setBPM(self, newBPM):
            self.samplesplate.setBPM(newBPM)
            return

    def setTala(self, name_cc, name_filler, NbeatCyclesPerCall):
            beatcyclesplate = BeatCyclesPlate(name_cc, 
                    name_filler, NbeatCyclesPerCall)
            self.beatsplate.setBeatCyclesPlate(beatcyclesplate)
            return

    def on_press(self,key):

            global store 
            global store2

            print (key)
            if key == keyboard.Key.space:
                # SamplesPlate.beatlen= 16000
                self.samplesplate.setBPM(self.currentBPM + 5)
                self.currentBPM += 5
                return False
            if key == keyboard.Key.backspace:
                self.samplesplate.setBPM(self.currentBPM - 5)
                self.currentBPM -= 5
                return False
            if key == keyboard.Key.tab:
                print ('You Stopped the Audio')
                self.stream.stop_stream()
                t = time.localtime()
                tempor = str(t)
                stringtemp = "finalarray" + tempor + ".csv"
                df = pd.DataFrame({'Index' : store3,'Desired BPM' : store4, 'PI_bpm': store, 'Raw_bpm': store2})
                df2 = pd.DataFrame({'Index' : self.x_record})
                df.to_csv("Sensor_BPM" + stringtemp, index=False)
                df2.to_csv("Sensor_Input"  + stringtemp, index=False)
                # stringtemp = "finalarray" + tempor + ".csv"
                # np.savetxt(stringtemp, store)
                # pdb.set_trace()
                # sys.exit(0)
                # self.thread_stream.stop()
                self.thread_stream.join()   
                self.sensor_input.join()
                self.estimating.join()
                self.sensor_input.stop()
                self.thread_stream.stop()
                self.estimating.stop()
                print(' stopped')
                return False
            # if key == keyboard.Key.ctrl_l:        
            #     self.putfiller=self.nbeats
            #     return False
            if key == keyboard.Key.caps_lock:
                if self.name_filler is None:
                    self.name_filler = "MridangaFiller"
                    self.beatcyclesplate.setnamefiller(self.name_filler)
                else:
                    self.name_filler = None
                    self.beatcyclesplate.setnamefiller(self.name_filler)
            if key == keyboard.Key.page_up:                 
                self.samplesplate.setBPM(self.currentBPM * 2)
                self.currentBPM = self.currentBPM * 2
                return False 
            if key == keyboard.Key.page_down:
                self.samplesplate.setBPM(self.currentBPM // 2)
                self.currentBPM = self.currentBPM // 2
                return False 

            return False    

    def estimatebeat(self,):

            global store
            global store2
            global store3   
            global store4   
            global bpm
            global indexing 
            # print("Estimate beat entered")
            self.currptr = self.leftptr + 196                                        # What is current pointer and what is left pointer
            if (self.leftptr ==0):                                                   # What are these arrangements
                for j in range(self.i):
                    if (self.y_record[j]==0):
                        self.currempty = self.currempty +1
                    else:
                        bpm = (11300 / (self.currempty)) + 1.2                       # bpm formula? 
                        self.currempty = 0
                        var1 = self.currentBPM - bpm                                 # Why var1 this is taken? 
                        var2 = bpm - self.currentBPM                                 # Current Error e[n]

                        if (var2 * self.esum <0):                                    # if q[n]*e[n] < 0
                            self.esum =0                                             #  Make Accumulated error q[n] = 0, if q[n]*e[n] < 0

                        if (var1 < (0.7 * self.currentBPM) and var1 > 0):            # Why this condition is taken?
                            error = (0.2)*var2 + (0.4) * self.esum                   # Control Input u[n] = kp*e[n] + ki*q[n] 
                            self.esum += var2                                        # Acuumulated error q[n] = q[n-1] + e[n]

                            if (self.currentBPM + error < bpm):                          # Why this is taken? 
                                self.currentBPM = ((3*bpm + self.currentBPM)/4)          # Why this is taken? 
                            else:           
                                self.currentBPM += (error)                               # Why this is taken? 

                            self.setBPM(self.currentBPM)    

                        elif (var2 < (0.7 * self.currentBPM) and var2 > 0):         # Why this condition is taken?
                            error = (0.2)*var2 + (0.4)* self.esum                   # Control Input u[n] = kp*e[n] + ki*q[n]
                            self.esum += var2                                       # Acuumulated error q[n] = q[n-1] + e[n]

                            if (self.currentBPM + error > bpm):                      # Why this is taken? 
                                self.currentBPM = ((3*bpm + self.currentBPM)/4)      # Why this is taken? 
                            else:
                                self.currentBPM += (error)                           # Why this is taken? 

                            self.setBPM(self.currentBPM)
                      
            else:                                                                   # What is this condition
                for j in range (self.currptr % 10000 , self.i % 10000):
                    if (self.y_record[j]==0):
                        self.currempty = self.currempty +1
                    else:
  
                        bpm = 11300 / (self.currempty) + 1.2                         # bpm formula? 
                        var1 = self.currentBPM - bpm                                 # Why var1 this is taken? 
                        var2 = bpm - self.currentBPM                                 # Current Error e[n]
                        self.currempty = 0                                           # Why this is taken? 

       
                        if (var2 * self.esum <0):                                    # if q[n]*e[n] < 0
                            self.esum =0                                             # Make Accumulated error q[n] = 0, if q[n]*e[n] < 0
                        if (var1 < (0.7 * self.currentBPM) and var1 >= 0):
        
                            error = (0.2)*var2 + (0.4) * self.esum                   # Control Input u[n] = kp*e[n] + ki*q[n]
                            self.esum += var2                                        # Acuumulated error q[n] = q[n-1] + e[n]

                            if (self.currentBPM + error < bpm):
                                self.currentBPM = ((3*bpm + self.currentBPM)/4)
                     
                            else:
                                self.currentBPM += (error)
                            self.setBPM(self.currentBPM)
                 
                        elif ( var2 < (0.7 * self.currentBPM) and var2 > 0):
                   
                            error = (0.2)*var2 + (0.4)*self.esum
                            self.esum += var2
                            if (self.currentBPM + error > bpm):
                                self.currentBPM = ((3*bpm + self.currentBPM)/4)
               
                            else:
                                self.currentBPM += (error)
                       
                            self.setBPM(self.currentBPM)
              

            if (bpm!=0) :
                store = np.append(store,self.currentBPM)
                store2 = np.append(store2, bpm)
                store3 = np.append(store3, indexing)        
                store4 = np.append(store4, 180) 
                indexing = indexing + 1     
            self.leftptr = self.leftptr + 4
    


    def sensorinput(self):
            global current_time1, elapsed_time   
                    
            while True:
                if serialInst.in_waiting:   
                    packet = serialInst.readline()  
                    reading= int(packet.decode('utf').rstrip('\n'))
                    print(reading)
                    self.x_record = np.append(self.x_record, reading)           
                    if (reading <= 5):              
                        # print("\n-------------- if (reading <= 5)---------1-")
                        self.y_record[self.i % 10000]=0 
                        self.prevreading = reading          
                        self.sum = 0            
                    elif (self.prevreading >5):
                        # print("-------------- elif (self.prevreading >5)-------2---")
                        current_time1 = time.perf_counter()   
                        # print(f"\nCurrent Tapped time {current_time1:.4f}")                                                                       
                        elapsed_time = current_time1 - self.last_key_press_time
                        self.last_key_press_time = current_time1                        
                        # print(f"\nTapped after previous tap {elapsed_time:.4f}")
                        self.y_record[self.i % 10000] = 0
                        self.prevreading = reading
                        self.sum = self.sum +1  
                    elif (self.prevreading <=5 ):
                        # print("-------------- elif (self.prevreading <=5 )--------3--")
                        self.y_record[self.i % 10000] = reading
                        self.prevreading = reading  
                        self.sum = self.sum +1  
                    if self.sum >= 500:      
                        # print("--------------self.sum >= 500-------4---")    
                        self.flag =1
                        self.sum =0
                    elif ((self.i) - self.leftptr) >= 200 :
                        # print("\n--------------elif ((self.i) - self.leftptr) >= 200-------5---") 
                        self.estimatebeat() 
                    self.i = self.i + 1
         

    def callback(self, in_data, frame_count, time_info, status): # !!!!!

            global serialInst   
            global sum              
                
            global obs
            global obsv
            'frame_count is 1024'   
            data = self.samplesplate.fetch(frame_count)
            return (data.astype(np.float32).tobytes(), pyaudio.paContinue)


if __name__ == "__main__":

    metronome = MetronomeGenerator()
sys.exit()

