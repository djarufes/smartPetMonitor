import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics  

from datetime import datetime 
import random 
import os 

print(tf.__version__)

audio_file_path='datasets/drinking_2.wav'
librosa_audio_data,librosa_sample_rate=librosa.load(audio_file_path)
print(librosa_audio_data)

plt.figure(figsize=(12, 4))
plt.plot(librosa_audio_data) 

wave_sample_rate, wave_audio = wav.read(audio_file_path)  


# Original audio in stereo
plt.figure(figsize=(12, 4))
plt.plot(wave_audio)

# MFCC Extraction begin here 
mfccs = librosa.feature.mfcc(y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=40)
print(mfccs.shape) 

audio_dataset_path='datasets'
metadata=pd.read_csv('metadata.csv')
metadata.head(50)

for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),str(row["file_name"]))
    print(file_name)

# Loop for extracting
extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),str(row["file_name"])+'.wav')
    print(file_name)
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels]) 

# converting extracted spectral features of drinking to Pandas dataframe
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head(50) 

# Split the dataset into independent and dependent dataset
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())
X.shape 


### MODEL TRAINING from here onwards ###
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))

### Train Test Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) 

### No of classes
num_labels=y.shape[1] 

model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam') 

# Trianing... 
num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='models/drinking_rec.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)

duration = datetime.now() - start
print("Training completed in time: ", duration)

test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print("Test Accuracy is: ", test_accuracy[1]) 

#Randomly select a file from validation set 
drink_or_eat = str(random.randint(0,1))
file_num = str(random.randint(1,22))
if drink_or_eat == '0': 
    string="drinking"
elif drink_or_eat == '1': 
    string ="eating"
testfile = string+"_"+file_num+".wav"

#File to be analyzed
filename='datasets/'+testfile
print("File to be analyzed by the model: "+filename,"\n")

#Feeding the audio data into the model
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
librosa_audio_data,librosa_sample_rate=librosa.load(filename)
plt.figure(figsize=(12, 4))
plt.plot(librosa_audio_data)

print("Extracted spectral feature array: \n")
print(mfccs_scaled_features)

mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
predicted_label=(model.predict(mfccs_scaled_features) > 0.5).astype("int32")
print("\nModel outcome Label: ", predicted_label)

#Interpreting the analyzed data from model 
if predicted_label[0][0] == 1: 
    final_behavior = "Drinking"
elif predicted_label[0][0] == 0: 
    final_behavior = "Eating"
print("\nFinal classified behavior: "+final_behavior)

def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features 
