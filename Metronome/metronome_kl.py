"""PyAudio Example: Play a wave file (callback version)"""
# https://stackoverflow.com/questions/20057306/creating-an-infinitely-long-sine-tone-in-python
# This version:
# Plays a beat cycle
'''
class SamplesPlate:
    self.plate  # np.array (N,)
    self.beatsplate   
    self.SAMPLERATE 
    def fetch(len) # get the buffer
    def refill():
        self.beatsplate.fetch(3)
    def keepFilled():
        while True:
            if len(self.plate)<3*len: self.refill

class BeatsPlate:
    # [[dha], [dhin, dhin], [dha], [-]]
    self.plate
    self.bpm
    self.beatcyclesplate 
    def fetch(Nbeats)
    def refill():
        self.beatcyclesplate.fetch(1)
class BeatCyclesPlate:
    # [C1, filler1, C1, C1, C1, filler1]
    self.plate  # array
    def fetch()
        if len(self.plate)<3: self.refill()
class CallCyclesPlate:
    # [P1, P1] 
    # P1 = [C1, C1, C1, filler1]
    self.plate
    def fetch():
        if len(self.plate)<3: self.refill
    def refill():
class Metronome():
    self.currentTala = [['tin'],['tin'],['dha'],['-']]
    self.currentBPM = 140
    callcyclesplate = CallCyclesPlate()
    def changeBPM(newBPM)
    def changeTala(newTala)
'''
DEBUG = True 

from queue import Queue
from re import X
import pyaudio, json 
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

serialInst = serial.Serial()
global store
global store2
store = np.empty((0), int)
store2 = np.empty((0), int)
store3 = np.empty((0), int)
store4 = np.empty((0), int)
# while True:
#     if serialInst.in_waiting:
#         packet = serialInst.readline()
#         reading= int(packet.decode('utf').rstrip('\n'))
#         change= reading - self.prevreading
#         if change==0:
#             sum=sum+1
#         else:
#             if sum==2:
#                 BPM=100
#             sum=0
#         self.prevreading= reading

# import Tkinter
# import tkMessageBox

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

# class BufferPlate:
#     def __init__(self, CHUNK) -> None:
#         self.plate = np.zeros(CHUNK * 10)
#         self.pointer_currentstroke2refill = 512 # start refill here # initial delay
#     def fetch(self, framelen): # get the buffer
#         self.framelen = framelen 
#         retval = self.plate[:framelen]*1
#         self.plate[:-framelen] = self.plate[framelen:]  # move the queue 
#         self.plate[-framelen:] *= 0
#         self.pointer_currentstroke2refill -= framelen  # refill pointer also moves
#         # self.keepFilled()
#         return retval
class SamplesPlate:
    def __init__(self, beatsplate, BPM, name_cc):
        self.dataio = dataIO(name_cc)
        self.plate = np.zeros(self.dataio.RATE * 5)  # np.array (N,)
        self.pointer_currentstroke2refill = 512 # start refill here # initial delay
        self.BPM = BPM       # can change dynamically
        self.stroke2wav = {}  # get wav from here
        self.beatsplate = beatsplate
        self.ON = True
        self.framelen = 100     # to activate refill in keepFilled
        self.beatlen = int(self.dataio.RATE * 60 / self.BPM)
        self.refill(1) # This cant be greater that beats in the beatcycle
        
    def fetch(self, framelen): # get the buffer
        self.framelen = framelen 
        retval = self.plate[:framelen]*1
        self.plate[:-framelen] = self.plate[framelen:]  # move the queue 
        self.plate[-framelen:] *= 0
        self.pointer_currentstroke2refill -= framelen  # refill pointer also moves
        self.keepFilled()
        return retval

    def refill(self, Nbeats):
        # global nbeats 
        'add more strokes to the plate'
        # self.beatlen = int(self.dataio.RATE * 60 / MetronomeGenerator.currentBPM)
        if DEBUG: 
            print('refill with BPM:', self.BPM)
            print('beatlen', self.beatlen)
            # store = np.append(store, self.BPM)
            # print("beatcycles", nbeats)
            
        RATE = self.dataio.RATE 
        beats = self.beatsplate.fetch(Nbeats) # [['tin'],['tin'],['-'],]
        if DEBUG: print('beats:', beats)
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
                if p<0: #++++++  trim non-causal part
                    wav = wav[-p:] 
                    p = 0
                try:
                    if DEBUG: 
                        print('p', p, 'len(wav)', len(wav), 'wav.shape', wav.shape) 
                    self.plate[p:p+len(wav)] += wav
                except ValueError:
                    self.extendPlate(p+len(wav))
                    # print(len(self.plate),p+len(wav),)
                    self.plate[p:p+len(wav)] += wav
                # try:
                #     if DEBUG: 
                #         print('p', p, 'len(wav)', len(wav), 'wav.shape', wav.shape) 
                #     self.plate[p+1024:p+1024+len(wav)] += wav
                # except ValueError:
                #     self.extendPlate(p+1024+len(wav))
                #     # print(len(self.plate),p+len(wav),)
                #     self.plate[p+1024:p+1024+len(wav)] += wav

                self.pointer_currentstroke2refill += strokelen # move pointer 
        return 

    def extendPlate(self, length):
        'extend the total length of plate to hold longer stroke wavs' 
        assert length > len(self.plate), (length, len(self.plate))
        self.plate = np.concatenate((self.plate, np.zeros(length-len(self.plate))))

    def keepFilled(self):
        # self.beatlen = int(self.dataio.RATE * 60 / MetronomeGenerator.currentBPM)
        Nbeats = 1      # CRUCIAL - the latency in changing BPM
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
        self.Nbeats = 3     # to activate refill in keepFilled
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
        if DEBUG: print('refill BeatsPlate')
        beats = self.beatcyclesplate.fetch()
        self.plate += beats
        return 

    def keepFilled(self):
        if self.ON:
            if len(self.plate)<self.Nbeats: 
                self.refill()
        return

    def setBeatCyclesPlate(self, beatcyclesplate):
        self.beatcyclesplate = beatcyclesplate
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
        self._NbeatsperBC = 3 # changed this, should be less than beats is any sequence
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

# if DEBUG:
#     bcp = BeatCyclesPlate('Mridanga(16)', 'MridangaFiller', 4)
#     bp = BeatsPlate(bcp)
#     sp = SamplesPlate(bp)
#     pdb.set_trace()
# y_record = np.empty((0), int)
# nbeats=0
# putfiller = 0
# x=0
# leftptr = 0
# currempty =0
# self.prevreading = 0
# sum=0
# flag =0

obs =0
obsv =0
bpm = 0
indexing = 0

class MetronomeGenerator():


    # sum = 0   
    def __init__(self):

        # currentBPM = 160
        self.i=0
        self.currentBPM= 60
        self.paused = False 
        self.bypass = 0
        self.hardware(self.bypass)
        # q=Queue()
        self.y_record = np.empty((10000), float)
        self.x_record = np.empty((0), int)
        # self.nbeats=0
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
        NbeatCyclesPerCall = 1

        self.Phi = np.array([[1,1],[0,1]])
        self.H = np.array([[1,0]])
        self.Q = np.eye(2) * 0.1     # tunable parameter
        self.R = 0.1                 # tunable parameter
        self.P_n = self.Q*1
        self.z_n = 1
        self.BPM_n = self.currentBPM
        self.x_n = np.array([[0,1]]).T 

        self.beatcyclesplate = BeatCyclesPlate(name_cc, self.name_filler, NbeatCyclesPerCall)
        x= self.beatcyclesplate._NbeatsperBC
        self.beatsplate = BeatsPlate(self.beatcyclesplate)
        self.samplesplate = SamplesPlate(self.beatsplate, self.currentBPM, name_cc)
        self.RATE = self.samplesplate.dataio.RATE
        # self.thread_fillsamplesplate = threading.Thread(target=self.samplesplate.keepFilled(), name='thread_fillsamplesplate')
        
        # CHUNK = 512
        # p = pyaudio.PyAudio()
        # self.stream = p.open(format=pyaudio.paFloat32,
        #                 channels=1,
        #                 rate=int(self.RATE),
        #                 input=True,
        #                 output=True,
        #                 frames_per_buffer=CHUNK,
        #                 stream_callback=self.callback)
        self.thread_stream = threading.Thread(target=self.streaming, name='thread_stream')
        self.sensor_input = threading.Thread(target=self.sensorinput, name = 'sensor_input')
        # self.estimating = threading.Thread(target=self.estimatebeat, name="estimating")
        self.thread_stream.start()
        self.sensor_input.start()
        # self.estimating.start()
        
        # while self.stream.is_active() or self.paused==True:
        #     with keyboard.Listener(on_press=self.on_press) as listener:
        #         listener.join()
        #     time.sleep(0.1)

        # self.stream.stop_stream()
        # self.stream.close()

        # p.terminate()
        
        # y_record = np.concatenate(y_record)
        # write("test.wav", self.RATE, q)
        
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
            # stringtemp = "finalarray" + tempor + ".csv"
            # df = pd.DataFrame({'Index' : store3,'Desired BPM' : store4, 'Array 1': store, 'Array 2': store2})
            # df2 = pd.DataFrame({'Index' : self.x_record})
            # df.to_csv(stringtemp, index=False)
            # df2.to_csv("x_record"  + stringtemp, index=False)
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
        # if key == keyboard.Key.alt_l:
        #     self.samplesplate.setBPM(60)
        #     self.currentBPM = 60
        #     return False
        return False

    def estimatebeat(self,):
        global store
        global store2
        global store3
        global store4
        global bpm  
        global indexing
        start = time.perf_counter()
        self.currptr = self.leftptr + 150
        if (self.leftptr ==0):
            for j in range(self.i):
                if (self.y_record[j]==0):
                    self.currempty = self.currempty +1
                else:       
                    bpm = (11300 / (self.currempty)) + 1.2  
                    self.currempty = 0  
                    self.K_n = self.Phi @ self.P_n @ self.Phi.T @ self.H.T / (self.H @ self.Phi @ self.P_n @ self.Phi.T @ self.H.T + self.R)
                    self.x_n = self.Phi @ self.x_n + self.K_n @ (self.z_n - self.H @ self.Phi @ self.x_n)       # x_n+1
                    self.P_n = (1 - self.K_n @ self.H) * (self.Phi @ self.P_n @ self.Phi.T + self.Q)   # P_n+1
                    self.z_n += 60/self.BPM_n       # z_n+1
                    self.BPM_n = bpm
                    print(self.x_n[1])
                    self.currentBPM = 60/self.x_n[1]
                    self.setBPM(self.currentBPM)
        else:
            for j in range (self.currptr % 10000 , self.i % 10000):
                if (self.y_record[j]==0):
                    self.currempty = self.currempty +1
                else:   
                    bpm = 11300 / (self.currempty) + 1.2
                    self.currempty = 0
                    self.K_n = self.Phi @ self.P_n @ self.Phi.T @ self.H.T / (self.H @ self.Phi @ self.P_n @ self.Phi.T @ self.H.T + self.R)
                    self.x_n = self.Phi @ self.x_n + self.K_n @ (self.z_n - self.H @ self.Phi @ self.x_n)       # x_n+1
                    self.P_n = (1 - self.K_n @ self.H) * (self.Phi @ self.P_n @ self.Phi.T + self.Q)   # P_n+1
                    self.z_n += 60/self.BPM_n       # z_n+1
                    self.BPM_n = bpm  
                    print("State O/P from Kalman filter i.e beat length",self.x_n[1])  
                    self.currentBPM = 60/self.x_n[1]
                    self.setBPM(self.currentBPM)
        if (bpm!=0):                   
            store = np.append(store,self.currentBPM)
            store2 = np.append(store2, bpm)
            store3 = np.append(store3, indexing)
            store4 = np.append(store4, 150)
            indexing = indexing + 1
        self.leftptr = self.leftptr + 50
        finish = time.perf_counter()
        temp = round((finish- start)*1000,2)
        # print("\t", f'Callback loop finished in {temp} ms')

                    
    def sensorinput(self):
        
        while True:
            if serialInst.in_waiting:
                packet = serialInst.readline()
                # start = time.perf_counter()
                reading= int(packet.decode('utf').rstrip('\n'))
                self.x_record = np.append(self.x_record, reading)
                # if (self.flag==0):
                if (reading == 0):
                    self.y_record[self.i % 10000]=0
                    # self.y_record = np.append(self.y_record, 0)
                    self.prevreading =0
                    self.sum = 0
                elif (self.prevreading !=0):
                    self.y_record[self.i % 10000] = 0
                    # self.y_record = np.append(self.y_record, 0)
                    self.prevreading = reading
                    self.sum = self.sum +1 
                elif (self.prevreading ==0):
                    self.y_record[self.i % 10000] = reading
                    # self.y_record= np.append(self.y_record, reading)
                    self.prevreading = reading
                    self.sum = self.sum +1 
                if self.sum >= 500:
                    self.flag =1
                    self.sum =0
                    # self.y_record = np.empty((0), int)
                    print(self.flag)
                elif ((self.i) - self.leftptr) >= 200 :
                # elif ((self.y_record.shape[0]) - self.leftptr) >= 20 :
                    # self.estimating.start()
                    # self.estimating.join()
                    # start = time.perf_counter()
                    self.estimatebeat()
                    # finish = time.perf_counter()
                    # temp = round((finish- start)*1000,2)
                    # print("\t", f'Callback loop finished in {temp} ms')
                self.i = self.i + 1
                # else:
                #     if (reading !=0):
                #         self.sum = self.sum +1 
                #     else:
                #         self.sum =0
                #     if self.sum >= 500:
                #         self.flag =0
                #         self.sum =0
                #         self.leftptr = 0
                #         self.i=0
                #         print(self.flag)
            # else:
            #     break
        # else:
        #     continue
                # finish = time.perf_counter()
                # temp = round((finish- start)*1000,2)
                # print("\t", f'Callback loop finished in {temp} ms')

    def callback(self, in_data, frame_count, time_info, status): # !!!!!

        global serialInst
        global sum
        global obs
        global obsv

        start = time.perf_counter()
        'frame_count is 1024'
        # self.samplesplate = SamplesPlate(self.beatsplate,1)
        # REC
        # y_record.append(in_data)
        # if len(y_record) >=1000:
        #     y_record = y_record[-1000:]
        # sum = 0
        # obs = 0
        data = self.samplesplate.fetch(frame_count)
        # obs += 1
        # sum = sum + 1
        # print(sum)
        # self.prevreading = 0
        #rd[self.i % 10000] = reading
        # y_record.append(in_data)
        # data = data.astype(np.float32).tobytes()
        # self.q.put(np.frombuffer(data, dtype=np.float32) for data in y_record))
        # return (data,pyaudio.paContinue)
        finish = time.perf_counter()
        temp = round((finish- start)*1000,2)
        # if (temp > 5 and obs <= 1000):
        #     sum += temp
        #     # obsv += 1
        #     print("\n")
        #     print("\t",f'Total time taken by Callback loop after 1000 callback calls is {float(sum)} ms')
        #     print("\n")
        # print("\t", f'Callback loop finished in {temp} ms')
        return (data.astype(np.float32).tobytes(), pyaudio.paContinue)


if __name__ == "__main__":
    # t1 = threading.Thread(target=MetronomeGenerator)
    # t2 = threading.Thread(target=sensorinput)
    metronome = MetronomeGenerator()
    

    # if False: # for getting wavfile
    #     outwav = []
    #     for i in range(100):
    #         outwav.append(metronome.samplesplate.fetch(CHUNK))
    #     outwav = np.concatenate(outwav)
    #     # librosa.output.write_wav('outwav.wav', np.concatenate(outwav), metronome.RATE, norm=False)
    #     sf.write('outwav.wav', outwav, metronome.RATE, 'PCM_24')
    # elif True:
    #     dataio = dataIO()
    #     # pdb.set_trace()
    # else:
    #     print("starting...")
    #     metronome.start()
    #     time.sleep(10)
    #     metronome.stop()
    #     print("stopped...")
    #     # aa = km.generate_callCycle()
    #     # plt.plot(np.arange(len(aa))/km.RATE, aa,'x-')
    #     # plt.show()

    #     # self.stream.stop_stream()
    #     # self.stream.close()
        # p.terminate()
sys.exit()

# # Audio synthesis
# class KirtanMetronome():
#     '''
#     stroke
#     beat
#     beatCycle
#     callCycle
#     '''
#     def __init__(self):
#         self.RATE = 8000
#         self.t = 0
#         self.beatCycle = [['tin'],['tin'],['dha'],['-']]
#         self.readStrokewavs() #'tin':[np.array([1,2,3]),1], 'dha':[np.array([4,5,6]),2], '-':[np.array([0]),0]}
#         self.bpm = 200
#         self.outwav = self.generate_callCycle() # this is finally played
#     # def readStroketable(self, stroketable='stroketable.csv'):
#     #     ''
#     #     return
#     def modifyBpm(self, newbpm):
#         'Call from Tkinter window'
#         self.bpm = newbpm
#         return
#     def readStrokewavs(self, strokesWav='strokeWavs.csv'):
#         'stroke,wav,delay_seconds'
#         strokes = set([stroke for beat in self.beatCycle for stroke in beat])
#         stroke2line = {}
#         with open(strokesWav,'r') as fid: # read csv
#             for line in fid:
#                 stroke2line[line.split(',')[0]] = line
#         self.stroke2wav = {}
#         for stroke in strokes: # read wav for only the strokes used in beatCycle
#             if stroke=='-':
#                 wavdata = None
#                 delay = 0
#             else:
#                 tokens = stroke2line[stroke].strip().split(',')
#                 wavdata, sr = librosa.load(tokens[1], sr=self.RATE, mono=False)
#                 if wavdata.ndim > 1:
#                     wavdata = wavdata[0] # one channel
#                 print(sr)
#                 delay = int(float(tokens[2])*self.RATE)
#             self.stroke2wav[stroke] = [wavdata, delay] 
#         return 
#     def circularAddNumpy(self, x1, x2, n0):
#         '''
#         modifies x1
#         x1[i] = x1[i] + x2[i-n0]
#         '''
#         assert len(x1)>len(x2), 'len(x1) should be larger'
#         i1_start = n0
#         i1_end = n0 + len(x2)
#         if i1_start<0:  # x2 crosses left end of x1
#             delta = -i1_start
#             x1[i1_start+len(x1):] = x1[i1_start+len(x1):] + x2[:delta]
#             x1[:i1_end] = x1[:i1_end] + x2[delta:]
#         elif i1_end>len(x1): # x2 crosses right end of x1
#             delta = len(x1)-i1_start
#             x1[i1_start:] = x1[i1_start:] + x2[:delta]
#             x1[:i1_end-len(x1)] = x1[:i1_end-len(x1)] + x2[delta:]
#         else:   # x2 is in the middle of x1
#             x1[i1_start:i1_end] = x1[i1_start:i1_end] + x2
#         return
#     def get_n0ofbeat(self, b):
#         'b can be float; max(b)=Nbeats'
#         return int(60./self.bpm * b * self.RATE) # 60 sec / 100 beats * Nbeats
#     def addStroke(self, data_callCycle, stroke, n0):
#         '''
#         data_callCycle is a np vector
#         n0 can be any int: centre of stroke should lie at this sample of data_callCycle
#         data_callCycle[i] += data_stroke[i-n0+delay_stroke]
#         '''
#         data_stroke, delay_stroke = self.stroke2wav[stroke]
#         if data_stroke is not None:
#             self.circularAddNumpy(data_callCycle, data_stroke, n0-delay_stroke)
#         return
#     def generate_callCycle(self):
#         'add a call cycle wav to a queue'
#         data_callCycle = np.zeros(self.get_n0ofbeat(len(self.beatCycle))) # sample index of 4th beat
#         for b,beat in enumerate(self.beatCycle):
#             for s,stroke in enumerate(beat):
#                 n0 = self.get_n0ofbeat(b + s/len(beat))
#                 self.addStroke(data_callCycle, stroke, n0)
#         return data_callCycle*0.1
#     def appendOutwav(self):
#         'generate one more call cycle and append to self.outwav'
#         self.outwav = np.hstack([self.outwav,self.generate_callCycle()])
#         return
#     def get_outwavbuffer(self, frame_count):
#         '''dequeues frame_count values from self.outwav
#         callback stops if len(retval)<frame_count
#         '''
#         retval = self.outwav[:frame_count]
#         self.outwav = self.outwav[frame_count:]
#         if len(self.outwav)<2*frame_count:
#             self.appendOutwav() # queue one more callCycle
#         return retval
#     def callback(self, in_data, frame_count, time_info, status):
#         'frame_count is 1024'
#         data = self.get_outwavbuffer(frame_count)
#         return (data.astype(np.float32).tostring(), pyaudio.paContinue)

# km = KirtanMetronome()

# # top = Tkinter.Tk()    

# # def increaseTempo():
# #    tkMessageBox.showinfo( "Hello Python", "Hello World")
    
# # B = Tkinter.Button(top, text ="Hello", command = helloCallBack)

# # B.pack()
# # top.mainloop()

# p = pyaudio.PyAudio()
# self.stream = p.open(format=pyaudio.paFloat32,
#                 channels=1,
#                 rate=int(km.RATE),
#                 output=True,
#                 stream_callback=km.callback)

# self.stream.start_stream()

# # while self.stream.is_active():
# #     time.sleep(0.1)
# aa = km.generate_callCycle()
# plt.plot(np.arange(len(aa))/km.RATE, aa,'x-')
# plt.show()

# self.stream.stop_stream()
# self.stream.close()
# p.terminate()
