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

checkpointer = ModelCheckpoint(filepath='models/drinking_rec.hdf5', 
                               verbose=1, save_best_only=True)

def audio_test(filename): 
    start = datetime.now()
    model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    test_accuracy=model.evaluate(X_test,y_test,verbose=0)
    print("Test Accuracy is: ", test_accuracy[1]) 

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
    return final_behavior 

def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features 
