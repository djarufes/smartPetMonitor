# smartPetMonitor
Smart Pet Monitoring I2I Project



## Audio Model 

**<u>Status</u>:** 

- Encountered some compatibility challenges. Some up-to-date libraries are not compatible with some other core libraries. Will manually downgrade these libraries to fix compatibility issues. 
- Aiming to develop an audio model for bird sound classification for practice. Sufficient data is provided by tensorflow guides, so it shouldn't be a big challenge. After we get familiar with tensorflow and other libraries, we will start developing our model from collected dataset. 
- More audio dataset collection is in progress. 



All libraries currently being used: 

```python
import tensorflow as tf
import tflite_model_maker as mm
from tflite_model_maker import audio_classifier
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import glob
import random

from IPython.display import Audio, Image
from scipy.io import wavfile
```



## Video Model 

