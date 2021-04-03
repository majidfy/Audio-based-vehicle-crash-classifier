import keras
from glob import glob
import numpy as np 
import pandas as pd 
from sklearn.utils import class_weight
from scipy.stats import mode
import tensorflow as tf
from utils import getGammatonegram, plot_confusion_matrix
    
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import re
import sys
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

###LOADING DATA FROM THE SEGMENTED AUDIO FILES AND THEIR LABELS
Path_MIVIA="D:/Audio_final_results/Training/" #Path to the segmented audio files


A_dir=glob(Path_MIVIA+"A/**/*.wav", recursive=True)
A_dir.sort(key=natural_keys)
B_dir=glob(Path_MIVIA+"B/**/*.wav", recursive=True)
B_dir.sort(key=natural_keys)
C_dir=glob(Path_MIVIA+"C/**/*.wav", recursive=True)
C_dir.sort(key=natural_keys)
D_dir=glob(Path_MIVIA+"D/**/*.wav", recursive=True)
D_dir.sort(key=natural_keys)

csv_file = open(Path_MIVIA+"A/labels.csv",'rb')
df = pd.read_csv(csv_file)
label_A=df.to_numpy()
label_A = np.insert(label_A, 0, 0, axis=0)

csv_file = open(Path_MIVIA+"B/labels.csv",'rb')
df = pd.read_csv(csv_file)
label_B=df.to_numpy()
label_B = np.insert(label_B, 0, 0, axis=0)

csv_file = open(Path_MIVIA+"C/labels.csv",'rb')
df = pd.read_csv(csv_file)
label_C=df.to_numpy()
label_C = np.insert(label_C, 0, 0, axis=0)
    
csv_file = open(Path_MIVIA+"D/labels.csv",'rb')
df = pd.read_csv(csv_file)
label_D=df.to_numpy()
label_D = np.insert(label_D, 0, 0, axis=0)


D1=[]
D2=[]
D3=[]
D4=[]

print("Creating Gammatonegrams..")
for row in range(len(A_dir)):
    gammatones, center_frequencies, sampling_rate, sound = getGammatonegram(A_dir[row])
    D1.append((gammatones,int(label_A[row])))
len(D1)

for row in range(len(B_dir)):
    gammatones, center_frequencies, sampling_rate, sound = getGammatonegram(B_dir[row])
    D2.append((gammatones,int(label_B[row])))
len(D2)

for row in range(len(C_dir)):
    gammatones, center_frequencies, sampling_rate, sound = getGammatonegram(C_dir[row])
    D3.append((gammatones,int(label_C[row])))
len(D3)

for row in range(len(D_dir)):
    gammatones, center_frequencies, sampling_rate, sound = getGammatonegram(D_dir[row])
    D4.append((gammatones,int(label_D[row])))
len(D4)
print("Gammatonegrams created")
#SAVE ALL THE CONSOLE OUTPUT IN A TEXT FILE
# sys.stdout = open("Results.txt", "w")

#RUN THE CODE IN A 4-FOLD SETTING USING THE 4 PARTITIONS CREATED EARLIER
for m in range(4):
    if m==0:
        A,B,C,D=D1,D2,D3,D4
    if m==1:
        A,B,C,D=D4,D1,D2,D3
    if m==2:
        A,B,C,D=D3,D4,D1,D2
    if m==3:
        A,B,C,D=D2,D3,D4,D1
    print('K fold is ', m)
    Train=A+B+C

    Test=D

    
    df1=pd.DataFrame(Train,columns=["ab","bc"])
    df1=df1.sample(frac=1)
    
    df2=pd.DataFrame(Test,columns=["ab","bc"])
    
    train=df1
    test=df2
    train_x=train["ab"].values
    test_x=test["ab"].values
    train_x = np.array([x.reshape( (64, 298, 1) ) for x in train_x])
    train_y=train["bc"].values
    test_x = np.array([x.reshape( (64, 298, 1) ) for x in test_x])
    test_y=test["bc"].values 
    
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(train_y),
                                                      train_y)
    
    class_weight_dict = {0: class_weights[0],
                    1: class_weights[1],
                    2: class_weights[2]}
    
    
    number_1=np.count_nonzero(train_y == 1)
    number_2=np.count_nonzero(train_y == 2)
    
  
    model=tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64,(3,3),input_shape=[64,298,1],activation="relu"),
        tf.keras.layers.MaxPooling2D(1,1),
          tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(16,(3,3),activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64,activation="relu"),      
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(3,activation="softmax"),])
    
    
    #compiling and fitting
    opt = keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy'])
    
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
    metrics=['accuracy'])
  
    history=model.fit(
        x=train_x, 
        y=train_y,
        epochs=60,
        batch_size=100,
        class_weight=class_weight_dict)
    score = model.evaluate(
        x=test_x,
        y=test_y)
    
    print('Test loss:', score[0])
    print('Test accuray:', score[1])
    # model.save('my_modelsounds_mivia.h5')
    
    #######PLOTTING CONFUSION MATRIX#######
    prediction= model.predict(test_x) .argmax(axis=1)
    cm = confusion_matrix(y_true=test_y, y_pred=prediction)
    cm_plot_labels = ['no crash','crash','skid']
    
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
    
    print(classification_report(test_y, prediction, target_names=cm_plot_labels))
    
    ########################EVENT BASED MODEL
    event_labels=np.zeros(len(test_y))
    event_labels_number=[]
    x=1
    for i in range(len(test_y)):
        if test_y[i]==1 or test_y[i]==2:
            event_labels[i]=x
            if test_y[i+1]==0:
                event_labels_number.append(int(test_y[i]))
                x=x+1
    
    missed=0 
    prediction_array=np.zeros((len(event_labels_number)))
    prediction_array_single_label=np.zeros((len(event_labels_number)))
    for i in range(x-1):
          Arr=  np.where(event_labels==i+1)
          m = mode(test_y[Arr])
          m=int(m[0])
          pred=prediction[Arr]
          ## prediction_array[i]=int(mode(pred)[0][0]) ##FOR MAX OF THE ARRAY
          if m==1:
              if 1 in pred:
                  prediction_array_single_label[i]=1
          elif m==2:
              if 2 in pred:
                  prediction_array_single_label[i]=2
                  
                  
          if 1 in pred or 2 in pred:
               crashes=np.count_nonzero(pred == 1)
               skids=np.count_nonzero(pred == 2)
               if crashes>=skids:
                   prediction_array[i]=1 # check if 2 or 3 present in the windows of the actual event. if yes, take the event most detected in those windows
               else:
                   prediction_array[i]=2
          else:
            missed=missed+1
            
        
    Arr=  np.where(event_labels==0)
    pred_nocrash=prediction[Arr]
    confusion=prediction_array==event_labels_number
    conf=prediction_array_single_label==event_labels_number
    counter=False
    FPR=0
    for z in range(len(pred_nocrash)):
        if pred_nocrash[z]!=0:
            if counter==True:
                FPR=FPR+1
                counter=False
            else:
                counter=True

    FPR=FPR/len(pred_nocrash)
    
    RR_paper=np.count_nonzero(conf == True)
    RR_paper=RR_paper/len(conf)
    
    RR=np.count_nonzero(confusion == True)
    RR=RR/len(confusion)
    print("RESULTS FOR THE EVENT BASED SYSTEM")
    print("FPR: ",FPR)
    print("RR_MEAN: ",RR)
    print("RR_SINGLE_LABEL: ",RR_paper)
    print('MISSED RATE',missed)
