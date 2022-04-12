% System Controller 

from audio_main import features_extractor 
from audio_test import audio_test, features_extractor 
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

import math
import os
import sys



path = '/datasets'
video_filename = "test_1.wav"


class AudioTools():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '\\' + filename
        
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 60 * 1000
        t2 = to_min * 60 * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '\\' + split_filename, format="wav")
        
    def multiple_split(self, min_per_split):
        total_mins = math.ceil(self.get_duration() / 60)
        for i in range(0, total_mins, min_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i+min_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_mins - min_per_split:
                print('All splited successfully')

    def convert_video_to_audio(video_file, output_ext="wav"):
        """Converts video to audio using MoviePy library
        that uses `ffmpeg` under the hood"""
        filename, ext = os.path.splitext(video_file)
        clip = VideoFileClip(video_file)
        clip.audio.write_audiofile(f"{filename}.{output_ext}")
        audio_filename = path + "{filename}.{output_ext}"
        return audio_filename 

