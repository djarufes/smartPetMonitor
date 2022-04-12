% System Controller 
from audio_main import features_extractor 
from audio_test import audio_test, features_extractor 
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

import csv 
import math
import os
import sys

path = '/datasets'
video_filename = "test_1.wav"



def main():
    audio_filename = convert_video_to_audio(video_filename)








def get_duration(self):
    return self.audio.duration_seconds

def single_split(self, from_sec, to_sec, split_filename):
    # t1 = from_min * 60 * 1000
    # t2 = to_min * 60 * 1000
    t1 = from_sec * 1000
    t2 = to_sec * 1000
    split_audio = self.audio[t1:t2]
    split_audio.export(self.folder + '\\' + split_filename, format="wav")
    
def multiple_split(self, sec_per_split):
    total_mins = math.ceil(self.get_duration() / 60)
    for i in range(0, total_mins, sec_per_split):
        split_fn = str(i) + '_' + self.filename
        self.single_split(i, i+sec_per_split, split_fn)
        print(str(i) + ' Done')
        if i == total_mins - sec_per_split:
            print('All splited successfully')

def convert_video_to_audio(video_file, output_ext="wav"):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood"""
    filename, ext = os.path.splitext(video_file)
    clip = VideoFileClip(video_file)
    clip.audio.write_audiofile(f"{filename}.{output_ext}")
    audio_filename = path + "{filename}.{output_ext}"
    return audio_filename 

def write_to_csv(data):
    with open('countries.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

if __name__ == "__main__":
    main()