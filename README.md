# Audio-based-vehicle-crash-classifier

This respository contains a classifer for detecting a vehicle crash and vehicle skid sounds using audio data.
We have used the MIVIA road audio events dataset, which can be found at https://mivia.unisa.it/datasets/audio-analysis/mivia-road-audio-events-data-set/.
We investigate 3 audio representative methods to implement the classifers on, namely; Spectrogram based, Gammatonnegram based and a time and frequency feature based method. 
The repository contains a script to segment the audio files from the dataset based on a specific window length and overlap percentage.
For the classifier implementation, we run a 4-Fold cross validation technique to evaluate the results obtained.
