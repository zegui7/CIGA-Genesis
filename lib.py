from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import AudioFileClip
from glob import glob
import time
import numpy as np
import time
import cv2
from moviepy.editor import VideoClip
import librosa

from multiprocessing import Queue,Process,SimpleQueue
from skimage.transform import AffineTransform
from skimage.transform import warp

def tempo_at_times(audio,sr,times):
    onset_env = librosa.onset.onset_strength(audio, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr,
                               aggregate=None)
    time_at_tempo = np.linspace(0,librosa.get_duration(audio,sr),num=tempo.shape[0])
    tempo_at_times = np.zeros(times.shape)
    tempo_bins = np.digitize(time_at_tempo,times)
    for x in range(0,tempo_bins[-1]):
        tempo_at_times[x] = np.mean(tempo[tempo_bins == x])
    
    return tempo_at_times

def average_intensity_every_n(audio,sr,n):
    m = sr / n
    audio = np.abs(audio)
    duration = int(np.ceil(audio.shape[0]/sr)*n)
    out = np.zeros([duration])
    for i in range(0,duration):
        a,b = int(i*m),int((i+1)*m)
        out[i] = np.mean(audio[a:b])
    out = (out - out.min()) / (out.max() - out.min())
    return out
    
def pad(array,size):
    if array.shape[0] > size[0]:
        diff = (array.shape[0] - size[0])//2
        a,b = diff,diff + size[0]
        array = array[a:b,:,:]
    if array.shape[1] > size[1]:
        diff = (array.shape[1] - size[1])//2
        a,b = diff,diff + size[1]
        array = array[:,a:b,:]
    n_channels = array.shape[2]
    output = np.zeros((*size,n_channels))
    sh = array.shape[:2]
    m_x = size[0] - sh[0]
    m_x = (m_x // 2, sh[0] + m_x // 2)
    m_y = size[1] - sh[1]
    m_y = (m_y // 2, sh[1] + m_y // 2)
    output[m_x[0]:m_x[1],m_y[0]:m_y[1],:] = array
    return output

class Transformer:
    def __init__(self):
        self.reset()
    
    def channel_switch(self,arr):
        if self.channels is not [0,1,2]:
            return arr[:,:,self.channels]
        return arr

    def invert_channels(self,arr):
        if len(self.channels_to_invert) != 0:
            for x in self.channels_to_invert:
                arr[:,:,x] = 255 - arr[:,:,x]
        return arr
    
    def set_changes(self):
        self.channels = [0,1,2]
        np.random.shuffle(self.channels)
        self.channels_to_invert = np.where(
            [np.random.uniform() > 0.5 for _ in range(3)])
        
    def transform(self,arr):
        for fn in [self.channel_switch,
                   self.invert_channels]:
            arr = fn(arr)
        return arr
    
    def reset(self):
        self.channels = [0,1,2]
        self.channels_to_invert = []

class Translator:
    def __init__(self,shape):
        self.shape = shape
        self.default = np.float32([[1, 0, 0], [0, 1, 0]])
        self.reset()
        
    def translate(self,arr):
        for ch,T in enumerate(self.T):
            if T is not self.default:
                arr[:,:,ch] = cv2.warpAffine(arr[:,:,ch], T,self.shape)
        return arr
    
    def set_changes(self):
        self.T = [
            np.float32([[1, 0, np.random.uniform(-self.shape[0]/8,
                                                 self.shape[0]/8)],
                        [0, 1, np.random.uniform(-self.shape[1]/8,
                                                 self.shape[1]/8)]])
            for x in range(3)]
        
    def reset(self):
        self.T = [
            np.float32([[1, 0, 0], [0, 1, 0]])
            for _ in range(3)]
        
class Frames:
    def __init__(self,
                 all_video_paths,
                 max_duration,
                 n_sec):
        self.all_video_paths = all_video_paths
        self.max_duration = max_duration
        self.n_sec = n_sec
        self.acc = 0.
        self.trigger_change()
    
    def generate(self,i):
        idx = self.acc
        if self.R_size < 4:
            self.trigger_change()
        if idx >= self.R_size or idx > self.max_duration:
            self.R = self.R[0:(idx-1)]
            self.R = np.flip(self.R)
            self.R_size = len(self.R)
            self.acc = 0
        j = self.R[self.acc]
        f = self.video.get_frame(j)
        self.acc += 1
        return f
    
    def trigger_change(self):
        try:
            self.video.close()
        except:
            pass
        i = np.random.randint(len(self.all_video_paths))
        self.video = VideoFileClip(self.all_video_paths[i])
        self.video_duration = self.video.duration
        self.fps = self.video.fps
        self.R = np.random.uniform(0,self.video_duration-self.max_duration/self.fps)
        M = self.video_duration - self.R - 2/self.fps
        self.R += np.linspace(0,M,num=self.fps*M - 2)
        self.R_size = len(self.R)
        self.acc = 0

class OnsetChecker:
    def __init__(self,onset_times):
        self.curr_L = 0
        self.onset_times = onset_times
        
    def update(self,t):
        onset = np.where(self.onset_times < t)[0]
        if len(onset) > self.curr_L:
            self.curr_L = len(onset)
            return True
        return False

class VideoFrameGenerator:
    def __init__(self,all_video_paths,
                 onset_times,shape,
                 max_duration,fps,
                 tempo_at_onset_times,
                 duration,
                 intensity_frame,
                 freak=True,
                 shift=True):
        
        self.shape = shape
        self.all_video_paths = all_video_paths
        self.max_duration = max_duration
        self.onset_times = onset_times
        self.fps = fps
        self.tempo_at_onset_times = tempo_at_onset_times
        self.duration = duration
        self.intensity_frame = intensity_frame
        self.freak = freak
        self.shift = shift
        self.n_sec = self.duration / self.fps
        
        self.frames = Frames(self.all_video_paths,
                             self.max_duration,
                             self.n_sec)
        self.onset_checker = OnsetChecker(self.onset_times)
        self.onset_checker_2 = OnsetChecker(self.onset_times)

        self.transformer = Transformer()
        self.translator = Translator((self.shape[1],self.shape[0]))

        self.first = True
        self.t = 0
        self.GO = True
        self.q = Queue(16)
        self.p = Process(target=self.bg,args=((self.q),))
        self.p.daemon = True
        self.p.start()
        
    def bg(self,q):
        t = 0
        changes = 0
        while self.GO == True:
            if self.onset_checker.update(t/self.fps) == True:
                self.frames.trigger_change()
                self.frames.max_duration = np.maximum(
                    2,
                    2*self.fps*1/(self.tempo_at_onset_times[changes]/60))
                changes += 1
                
            t = int(t)
            try:
                intensity = self.intensity_frame[t]
            except:
                intensity = 0
            f = self.frames.generate(t)
            f = f.astype(np.int16) - intensity
            f = np.clip(f,0,255).astype(np.uint8)
            q.put([f,t])
            t += 1
                        
            if t > (self.fps * self.duration + 5):
                self.GO = False
        q.put(None)
    
    def get(self,t):
        if self.first == True:
            self.first = False
            return np.zeros((*self.shape,3))
        f,t = self.q.get()
        if self.onset_checker_2.update(t/self.fps) == True:
            if self.freak == True:
                if np.random.uniform() > 0.5:
                    self.transformer.set_changes()
                else:
                    self.transformer.reset()
            if self.shift == True:
                if np.random.uniform() > 0.5:
                    self.translator.set_changes()
                else:
                    self.translator.reset()
        if self.freak == True:
            f = self.transformer.transform(f)
        f = pad(f,self.shape)
        if self.shift == True:
            f = self.translator.translate(f)
        return f
    
    def empty_queue(self):
        do = True
        while do == True:
            x = self.q.get()
            if x is None:
                do = False
        self.q.close()
