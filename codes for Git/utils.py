
import xml.etree.ElementTree as ET
import numpy as np
import librosa

import itertools
import matplotlib.pyplot as plt
from gtg import gammatonegram
from gtg_example import gtgplot
import scipy.io.wavfile as wf

def getGammatonegram(audio_file, log_constant=1e-80, dB_threshold=-50.0):
    """ Convert sound into gammatonegram, with amplitude in decibels"""
    sampling_rate, sound = wf.read(audio_file)
    #sxx, center_frequencies = getGammatonegram(sound, sampling_rate)
    sxx, center_frequencies = gammatonegram(sound, sr=sampling_rate, fmin=20, fmax=int(sampling_rate/2.))
    sxx[sxx == 0] = log_constant
    sxx = 20.*np.log10(sxx) #convert to dB
    sxx[sxx < dB_threshold] = dB_threshold  
    return sxx, center_frequencies, sampling_rate, sound


def read_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()
    total_events=len(root[0].getchildren())
    table=np.zeros((total_events,4))
    for i in range(total_events):
        if root[0][i][1].text=='2':
            table[i,0]=2
        else:
            table[i,0]=1
        table[i,1]=root[0][i][3].text
        table[i,2]=root[0][i][4].text
        table[i,3]=((table[i,2]-table[i,1])/2)+table[i,1]
    return(table)

def make_labels(my_table, window_len, shift,total_windows):
    x=0    
    starting=np.linspace(0,(total_windows-1)*shift, total_windows)
    ending=np.linspace(window_len,(total_windows-1)*shift+window_len, total_windows)    
    label=np.zeros((total_windows))
    i=0 
    
    while (x<len(my_table)):
        s_p=my_table[x,1]
        e_p=my_table[x,2]
        label_event=my_table[x,0]
        if  ending[i]<s_p :
            label[i]=0
            
        elif ending[i]>s_p and (ending[i]-s_p)<0.15*(e_p-s_p) or (e_p-starting[i])<0.5*(e_p-s_p):
            label[i]=5
        
        elif ending[i]>s_p and (ending[i]-s_p)>0.15*(e_p-s_p) and (e_p-starting[i])>0.5*(e_p-s_p):
            label[i]=label_event
        if starting[i]>=e_p:
            label[i]=0
            x=x+1
        i=i+1
    return label

def plot_confusion_matrix(cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    

def make_spec(ps):  
        y,sr = librosa.load(ps)
        spec=librosa.feature.melspectrogram(y=y,sr=sr)
        spec=spec.reshape((1,128,130,1))
        return spec