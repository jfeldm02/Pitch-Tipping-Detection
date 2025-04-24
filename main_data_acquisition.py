import pose_estimation
import data_preprocessing
import json
import pandas as pd
import numpy as np
from scipy.signal import medfilt
import os
import tqdm
import cv2
import mediapipe as mp
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor, as_completed

'''
README:

This file is the main data acquisition function. It assesses your unique pitcher's video samples, runs MediaPipe pose analysis on them,
preprocesses the acquired data, and saves both the raw data and processed data into a json file path.

Inputs: See below
Outputs: raw_data_output.json, processed_data_output.json

Please note the following:
- Files were based on 60 FPS broadcast samples and read a maximum of 300 FPS. If your pitcher throws after 300/60 = 5 seconds,
either clip the videos prior to use or investigate the associated pose_estimation.py python files downloaded alongside this code. 
- Refer to the video_cropping_calibration.ipynb file to find good calibration variable values. This impacts data quality significantly. 

* Apple M3 Pro using 8 cores took 45 minutes to complete 1100 video samples. This code automatically uses 4 less than your computer's maximum cores. 
'''

# User Variables:
# pitcher_name = 'Bello_Brayan' # This variable isn't acually used but good to keep track of in your file name paths
pitcher_height = 72 # pitcher's height in inches necessary for data scaling
pitcher_handedness_right = True # True if the pitcher throws with his right hand 
# Folder with subfolders labeled by pitcher's pitch containing all organized video samples 
main_directory = '.../your_path_sample_pitcher'
# Raw ModelPipe human pose estimation data saved here as a JSON file. Useful for separate potential future analyses of data 
raw_data_output = '.../your_path_unprocessed.json'
# Processed ModelPipe human pose estimation data saved here as a JSON file. Necessary for this project's models
processed_data_output = '.../your_path_processed.json'
# Use same video as used in video_cropping_calibration.ipynb
good_video_sample = '.../your_path.mp4'
x_percent = 0.25 # Video cropping variable. Copy in from video_cropping_calibration.ipynb
y_percent = 0.3 # Video cropping variable. Copy in from video_cropping_calibration.ipynb 
width_percent = 0.4 # Video cropping variable. Copy in from video_cropping_calibration.ipynb
height_percent = 0.6 # Video cropping variable. Copy in from video_cropping_calibration.ipynb 

def main_data_acquisition(pitcher_height, pitcher_handedness_right, main_directory, raw_data_output, processed_data_output, 
                          good_video_sample, x_percent, y_percent, width_percent, height_percent):
    # Outputs raw MediaPose data file
    pose_estimation.process_videos_to_json(
        base_directory=main_directory,
        output_json_path=raw_data_output,
        template_video_path=good_video_sample,
        x_percent= x_percent,
        y_percent= y_percent,
        width_percent= width_percent,
        height_percent= height_percent
        )
    print(f"Raw data processed and saved to: {raw_data_output}")

    # Outputs processed MediaPose data file. Also outputs dataframes locally if user wants to add their own exploratory code.
    dataframes_unprocessed, dataframes_processed = data_preprocessing.data_processing(
        json_file_path= raw_data_output,
        pitcher_height= pitcher_height, 
        pitcher_handedness_right= pitcher_handedness_right,
        processed_output_path=processed_data_output
        )
    print(f"Pre-processed data processed and saved to: {processed_data_output}")

    return dataframes_unprocessed, dataframes_processed

if __name__ == "__main__":
    dataframes_unprocessed, dataframes_processed = main_data_acquisition(
        pitcher_height=pitcher_height,
        pitcher_handedness_right=pitcher_handedness_right,
        main_directory=main_directory,
        raw_data_output=raw_data_output,
        processed_data_output=processed_data_output,
        good_video_sample=good_video_sample,
        x_percent=x_percent,
        y_percent=y_percent,
        width_percent=width_percent,
        height_percent=height_percent
    )
    
    