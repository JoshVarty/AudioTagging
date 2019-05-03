import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Audio
import librosa
import torch
from fastai.basics import ItemBase 

# Parent classes used to distinguish transforms for data augmentation and transforms to convert audio into image
class MyDataAugmentation:
    pass

class MySoundToImage:
    pass

class AudioClip(ItemBase):
    def __init__(self, signal, sample_rate, fn):
        self.data = signal # Contains original signal to start 
        self.original_signal = signal.clone()
        self.processed_signal = signal.clone()
        self.sample_rate = sample_rate
        self.fn = fn

    def __str__(self):
        return '(duration={}s, sample_rate={:.1f}KHz)'.format(
            self.duration, self.sample_rate/1000)

    def clone(self):
        return self.__class__(self.data.clone(), self.sample_rate, self.fn)

    def apply_tfms(self, tfms, **kwargs):
        for tfm in tfms:
            self.data = tfm(self.data, **kwargs)
            if issubclass(type(tfm), MyDataAugmentation):
                self.processed_signal = self.data.clone().cpu()
        return self
    
    @property
    def num_samples(self):
        return len(self.data)

    @property
    def duration(self):
        return self.num_samples / self.sample_rate

    def show(self, ax=None, figsize=(5, 1), player=True, title=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        if title:
            ax.set_title("Class: " + str(title) + " \nfilename: " + str(self.fn))
        
        timesteps = np.arange(self.original_signal.shape[1]) / self.sample_rate
        
        ax.plot(timesteps, self.original_signal[0]) 
        if self.original_signal.size(0) > 1: # Check if mono or stereo
            ax.plot(timesteps, self.original_signal[1]) 
        ax.set_xlabel('Original Signal Time (s)')
        plt.show()
        
        timesteps = np.arange(self.processed_signal.shape[1]) / self.sample_rate

        _, ax = plt.subplots(figsize=figsize)
        if title:
            ax.set_title("Class: " + str(title) + " \nfilename: " + str(self.fn))
        ax.plot(timesteps, self.processed_signal[0]) 
        if self.processed_signal.size(0) > 1: # Check if mono or stereo
            ax.plot(timesteps, self.processed_signal[1]) 
        ax.set_xlabel('Processed Signal Time (s)')
        plt.show()
        
        if player:
            # unable to display an IPython 'Audio' player in plt axes
            display("Original signal")
            display(Audio(self.original_signal, rate=self.sample_rate))
            display("Processed signal")
            display(Audio(self.processed_signal, rate=self.sample_rate))
