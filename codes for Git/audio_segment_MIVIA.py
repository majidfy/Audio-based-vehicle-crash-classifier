
from glob import glob
import numpy as np 
import math
from pydub import AudioSegment
import soundfile as sf
from utils import read_xml, make_labels

#SPECIFY VARIABLES
dataset=['A','B','C','D']
path_dataset="D:/PRP Project/Audio Processing/Dataset/MIVIA/"
destination_path="D:/Audio_final_results/Training/"
window_len=3
shift=0.3
    
for i in range(len(dataset)):
    Xml_dir=glob(path_dataset+dataset[i]+"/**/*.xml", recursive=True)
    final_label=[]
    file_name=[]
    A=0
    Wav_dir=glob(path_dataset+dataset[i]+"/v2/**/*.wav", recursive=True)
    for j in range(len(Xml_dir)):
        f = sf.SoundFile(Wav_dir[j])
        duration=(len(f) / f.samplerate)

        total_windows=math.floor(((duration-window_len)/shift)+1)    
        start=np.linspace(0,(total_windows-1)*shift, total_windows)
        end=np.linspace(window_len,(total_windows-1)*shift+window_len, total_windows)
        newAudio = AudioSegment.from_wav(Wav_dir[j])
        my_table=read_xml(Xml_dir[j])
        ########Load model
        label=make_labels(my_table, window_len, shift,total_windows)
        
        
        for k in range(len(start)):
            if label[k]!=5:
                sampleaudio = newAudio[start[k]*1000:end[k]*1000]
                path=destination_path +dataset[i]+'/'+str(A)+'.wav'
                A=A+1
                file_name.append(str(j)+str(k))
                sampleaudio.export(path, format="wav") #Exports to a wav file in the current path.
        AA=np.where(label==5)        
        new_a=np.delete(label,AA)        
        final_label.append(new_a)
    np_label=np.hstack(final_label[:]) 
    np.savetxt(destination_path+dataset[i]+'/labels.csv', np_label, delimiter=",")
    


