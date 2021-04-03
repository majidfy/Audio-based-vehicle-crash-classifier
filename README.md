# Audio-based-vehicle-crash-classifier

This respository contains a classifer for detecting a vehicle crash and vehicle skid sounds using audio data.
We have used the MIVIA road audio events dataset, which can be found at https://mivia.unisa.it/datasets/audio-analysis/mivia-road-audio-events-data-set/.
We investigate 3 audio representative methods to implement the classifers on, namely; Spectrogram based, Gammatonnegram based and a time and frequency feature based method. 
The repository contains a script to segment the audio files from the dataset based on a specific window length and overlap percentage.
For the classifier implementation, we run a 4-Fold cross validation technique to evaluate the results obtained. There are 2 evaluation methods that we consider; Frame and Event based. In the frame based, the evaluation is done on each frame seperately and the results are reported. For the event based, a group of frames which correspond to a single event are clubed together and a single prediction is made based on the classifer results for all those frames
