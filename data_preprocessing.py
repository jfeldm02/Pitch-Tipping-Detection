import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import medfilt
import warnings

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

def json_to_dataframe(data, video_index):
    """
    Convert landmark data from a JSON file to a pandas DataFrame for a specific video.
    
    Parameters:
    json_file_path (str): Path to the JSON file
    video_index (int): Index of the video to extract (default: 0 for first video)
    
    Returns:
    pd.DataFrame: DataFrame with all landmark data across all frames
    """
    
    # Make sure the video index is valid
    if video_index >= len(data['videos']):
        raise ValueError(f"Video index {video_index} is out of range (total videos: {len(data['videos'])})")
    
    # Get the video data
    video = data['videos'][video_index]
    landmark_data = video['landmark_data']
    landmark_names = data['metadata']['landmark_names']
    
    # Determine how many frames we have by finding the length of any landmark's data list
    # Use the first landmark that has data as a reference
    for landmark_idx in landmark_data:
        if landmark_data[landmark_idx]:  # Check if the list is not empty
            num_frames = len(landmark_data[landmark_idx])
            break
    else:
        # If no landmarks have data, return an empty DataFrame with the correct columns
        return pd.DataFrame(columns=['frame'] + 
                            [f'{name}_{suffix}' for name in landmark_names for suffix in ['x', 'y', 'z', 'presence']] +
                            ['filename', 'pitch_type'])
    
    # Prepare data for DataFrame
    rows = []
    
    for frame_idx in range(num_frames):
        # Start with the frame index
        row = {'frame': frame_idx}
        
        # Add data for each landmark
        for landmark_idx_str, landmark_frames in landmark_data.items():
            landmark_idx = int(landmark_idx_str)
            
            if landmark_idx < len(landmark_names) and frame_idx < len(landmark_frames):
                landmark_name = landmark_names[landmark_idx]
                lm = landmark_frames[frame_idx]
                
                if len(lm) >= 4:  # Make sure we have all x, y, z, presence values
                    row[f'{landmark_name}_x'] = lm[0]  # x
                    row[f'{landmark_name}_y'] = lm[1]  # y
                    #row[f'{landmark_name}_z'] = lm[2]  # z, Preprocessing step a-priori to avoid unnecessary computations
                    row[f'{landmark_name}_presence'] = lm[3]  # visibility
                else:
                    # If landmark data is incomplete, add NaN values
                    row[f'{landmark_name}_x'] = None
                    row[f'{landmark_name}_y'] = None
                    row[f'{landmark_name}_z'] = None
                    row[f'{landmark_name}_presence'] = None
            else:
                # If landmark is missing or out of range, continue to next landmark
                continue
        
        rows.append(row)
    
    # Create DataFrame
    dataframe = pd.DataFrame(rows)
    
    # Add video metadata
    dataframe['filename'] = video['filename']
    dataframe['pitch_type'] = video['pitch_type']
    
    return dataframe

########################################################################################################################################################
########################################################################################################################################################

def data_processing(json_file_path, pitcher_height, pitcher_handedness_right, processed_output_path):
    '''
    Generates a list of dataframes, one per video sample, with the following preprocessing adjustments:
        -   Averages certain landmarks into common groups to smooth data and maintain full representation 
        -   Drops z-coordinates due to lack of reliance from a 2D video sample 
        -   Sets a consistent reference point so that all videos have consistent coordinates relative to eachother 
        -   Scales all coordinates from pixel location on the original frame to inches based on the pitcher's height
        -   Uses pitcher's consistent throwing hand kick as consistent reference point to gauge frame clipping relative
            to user chosen pitch tipping end. Equalizes length of every sample
        -   Uses every 4th frame sample to reduce inputs and smooth data. Still a sound representation of data.

    Parameters:
    pitcher_json_file_path (str): Output from pose estimation processing function.
    pitcher_height (int): Pitcher height in inches.
    pitcher_handedness_right (bool): True if the pitcher is right handed. False if the pitcher is left-handed.
    processed_output_path (str): Output to directory 

    Returns:
    dataframes_processed: A list of processed dataframes consisting of all landmark coordinates across all videos. 
                        | dataframe | pitch_type | video_index | video file name |
    dataframes_unprocessed: A list of preprocessed dataframes with same attributes. For preservation of original data for future tinkering
    '''
        # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Determine handedness
    pitcher_handedness = 'right' if pitcher_handedness_right else 'left'
    
    # List of all landmarks
    landmarks = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", 
                "right_eye_inner", "right_eye", "right_eye_outer", 
                "left_ear", "right_ear", "mouth_left", "mouth_right", 
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
                "left_wrist", "right_wrist", "left_pinky", "right_pinky", 
                "left_index", "right_index", "left_thumb", "right_thumb", 
                "left_hip", "right_hip", "left_knee", "right_knee", 
                "left_ankle", "right_ankle", "left_heel", "right_heel", 
                "left_foot_index", "right_foot_index"]
    
    dataframes_unprocessed = []
    dataframes_processed = []
    
    for video in tqdm(range(len(data['videos'])), desc="Processing Videos", unit="video"):
        # Checks if the video has a landmark dataframe
        if not data['videos'][video]['landmark_data']:
            continue

        # Load individual video dataframe
        dataframe_original = json_to_dataframe(data=data, video_index=video)

        # Store a copy in unprocessed dataframes
        dataframes_unprocessed.append({
            'dataframe': dataframe_original,
            'pitch_type': dataframe_original['pitch_type'].iloc[0],
            'video_index': video,
            'filename': dataframe_original['filename'].iloc[0]
        })
        
        if dataframe_original.empty:
            print(f"Video {video} is empty, skipping.")
            continue
                
        # Check if pitch_type exists
        if 'pitch_type' not in dataframe_original.columns or dataframe_original['pitch_type'].empty:
            print(f"Video {video} has no pitch_type data, skipping.")
            continue
        
        # Filter columns to keep relevant landmarks
        keep_columns = ['frame']
        for landmark in landmarks:
            keep_columns.extend([f'{landmark}_x', f'{landmark}_y', f'{landmark}_presence'])

        dataframe = dataframe_original[keep_columns]
        
        # Get reference points and calculate scale
        ankle_x = dataframe.loc[0, f'{pitcher_handedness}_ankle_x']
        ankle_y = dataframe.loc[0, f'{pitcher_handedness}_ankle_y']
        ground = dataframe.loc[0, f'{pitcher_handedness}_heel_y']
        pitcher_nose = dataframe.loc[0, 'nose_y']
        
        pixel_height = pitcher_nose - ground
        actual_height = pitcher_height - 4  # Assume 4 inches between top of head and nose
        scale = actual_height / pixel_height
            
        # Apply scaling to all values
        for landmark in landmarks:
            dataframe[f'{landmark}_x'] = scale * (dataframe[f'{landmark}_x'] - ankle_x)
            dataframe[f'{landmark}_y'] = scale * (dataframe[f'{landmark}_y'] - ankle_y)

        # Define groups of landmarks to average
        face_landmarks = ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 
                        'right_eye_inner', 'right_eye', 'right_eye_outer', 
                        'left_ear', 'right_ear', 'mouth_left', 'mouth_right']

        left_hand_landmarks = ['left_wrist', 'left_pinky', 'left_index', 'left_thumb']
        right_hand_landmarks = ['right_wrist', 'right_pinky', 'right_index', 'right_thumb']

        left_foot_landmarks = ['left_ankle', 'left_heel', 'left_foot_index']
        right_foot_landmarks = ['right_ankle', 'right_heel', 'right_foot_index']

        # Groups to process
        landmark_groups = [
            ('face', face_landmarks),
            ('left_hand', left_hand_landmarks),
            ('right_hand', right_hand_landmarks),
            ('left_foot', left_foot_landmarks),
            ('right_foot', right_foot_landmarks)
        ]

        # Process each group - average landmarks
        for group_name, group_landmarks in landmark_groups:
            # Initialize empty lists
            x_values = []
            y_values = []
            presence_values = []
            
            # For each frame
            for frame in dataframe['frame'].unique():
                frame_data = dataframe[dataframe['frame'] == frame]
                
                # Collect values for each landmark in this group
                x_vals = []
                y_vals = []
                presence_vals = []
                
                for landmark in group_landmarks:
                    x_col = f"{landmark}_x"
                    y_col = f"{landmark}_y"
                    presence_col = f"{landmark}_presence"
                    
                    if x_col in frame_data.columns and y_col in frame_data.columns and presence_col in frame_data.columns:
                        x_val = frame_data[x_col].values[0]
                        y_val = frame_data[y_col].values[0]
                        presence_val = frame_data[presence_col].values[0]
                        
                        # Only include non-NaN values
                        if not np.isnan(x_val) and not np.isnan(y_val) and not np.isnan(presence_val):
                            x_vals.append(x_val)
                            y_vals.append(y_val)
                            presence_vals.append(presence_val)
                
                # Calculate average if we have valid values
                if len(x_vals) > 0 and len(y_vals) > 0 and len(presence_vals) > 0:
                    x_values.append(np.mean(x_vals))
                    y_values.append(np.mean(y_vals))
                    presence_values.append(np.mean(presence_vals))
                else:
                    x_values.append(np.nan)
                    y_values.append(np.nan)
                    presence_values.append(np.nan)
            
            # Add the new averaged landmarks to the dataframe
            dataframe[f"{group_name}_x"] = x_values
            dataframe[f"{group_name}_y"] = y_values
            dataframe[f"{group_name}_presence"] = presence_values
        
        # Get foot data for time alignment
        if pitcher_handedness_right:
            y_data_foot = dataframe['right_foot_y'].values.copy()
        else:
            y_data_foot = dataframe['left_foot_y'].values.copy()
                
        # Apply median filter
        kernel_size = 5
        smoothed_y = medfilt(y_data_foot, kernel_size)
        
        # Focus on frames 100-300
        start_frame = 100
        end_frame = min(300, len(smoothed_y))
        
        if end_frame > start_frame:
            # Find max in the relevant range
            focus_range = smoothed_y[start_frame:end_frame]
            max_idx_in_range = np.argmax(focus_range)
            max_y_value_frame = start_frame + max_idx_in_range

            frame_range_max = max_y_value_frame - 50
            frame_range_min = frame_range_max - 100

            if frame_range_min < 0:
                # Handle negative frame range by adding padding frames
                first_frame = dataframe.iloc[0].copy()
        
                # Create new rows for each missing frame
                for frame in range(0, frame_range_min + 1):
                    new_row = first_frame.copy()
                    new_row['frame'] = frame
                    
                    # Set presence score to 0
                    for col in dataframe.columns:
                        if col.endswith('_presence'):
                            new_row[col] = 0
                    
                    # Add the new row
                    dataframe = pd.concat([pd.DataFrame([new_row]), dataframe], ignore_index=True)
                
                # Reset frame_range_min to 0
                frame_range_min = 0
                
            # Filter dataframe to the selected frame range
            filtered_dataframe = dataframe[(dataframe['frame'] >= frame_range_min) & 
                                 (dataframe['frame'] <= frame_range_max)]
            
            # Keep every 4th frame
            #dataframe = dataframe[dataframe['frame'] % 4 == 0]
            total_frames = len(filtered_dataframe)
            indices_to_keep = np.linspace(0, total_frames-1, 25, dtype=int)
            dataframe = filtered_dataframe.iloc[indices_to_keep].reset_index(drop=True)
        
        # Drop individual landmark columns
        columns_to_drop = ['frame']
        for group_name, group_landmarks in landmark_groups:
            for landmark in group_landmarks:
                columns_to_drop.extend([f"{landmark}_x", f"{landmark}_y", f"{landmark}_presence"])
       
        dataframe = dataframe.drop(columns=columns_to_drop)
        
        # Add to processed dataframes
        dataframes_processed.append({
            'dataframe': dataframe,
            'pitch_type': dataframe_original['pitch_type'].iloc[0],
            'video_index': video,
            'filename': dataframe_original['filename'].iloc[0]
        })

    # Save the processed data to JSON
    if processed_output_path and dataframes_processed:
        processed_json = []
        for item in dataframes_processed:
            # Convert DataFrame to dict for JSON serialization
            df_dict = item['dataframe'].to_dict(orient='records')
            processed_json.append({
                'dataframe': df_dict,
                'pitch_type': item['pitch_type'],
                'video_index': item['video_index'],
                'filename': item['filename']
            })
        
        with open(processed_output_path, 'w') as f:
            json.dump(processed_json, f, indent=2)
        print(f"Saved processed data for {len(dataframes_processed)} videos to {processed_output_path}")
    
    return dataframes_unprocessed, dataframes_processed
    