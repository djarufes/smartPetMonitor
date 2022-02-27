# Scripts 

**main.py** : main where we will run the model and stuff 

**EDA.ipynb** : EDA script for metadata and sorting 

**Pre-processing.ipynb** : Data pre-processing 

**Training.ipynb** : training dataset 

## Testing Script 

```python
#Randomly select a file from validation set 
import random
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
```





## Metadata class

1 - Drinking 

2 - Eating 

3 - Misc

# Resources 

https://www.tensorflow.org/tutorials/audio/transfer_learning_audio 

https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/tutorials/model_maker_audio_classification.ipynb#scrollTo=wbMc4vHjaYdQ

https://developer.apple.com/videos/play/wwdc2019/425/ 



Live: https://www.youtube.com/watch?v=8-vl9bNY9aI 

https://github.com/ShawnHymel/tflite-speech-recognition 

Librosa Loading : https://www.youtube.com/watch?v=CtiBW8T80SY 

