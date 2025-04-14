import os
import json
import tqdm
import cv2
import mediapipe as mp
import numpy as np
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor, as_completed


def extract_template_from_video(video_path):
    """
    Extract the first frame of a video to use as a template.
    
    Parameters:
    video_path (str): Path to the template video
    
    Returns:
    frame (np.ndarray): First frame of the video as template image, or None if extraction fails
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open template video file: {video_path}")
        return None
    
    # Read the first frame
    success, frame = cap.read()
    cap.release()
    
    if not success:
        print(f"Error: Could not read first frame from template video: {video_path}")
        return None
    
    return frame

########################################################################################################################################################
########################################################################################################################################################

def detect_pitcher(frame, template_img, similarity_threshold=0.25):
    """
    Detect if a pitcher is present in a frame by comparing it to a template image.
    This replaces the landmark-based detection with pure image similarity.
    
    Parameters:
    frame (np.ndarray): The frame to analyze
    template_img (np.ndarray): Template image containing a pitcher
    similarity_threshold (float): Threshold for similarity (0-1)
    
    Returns:
    bool: True if the frame is similar to the template (pitcher present), False otherwise
    float: Similarity score for debugging
    """
    if template_img is None:
        raise ValueError("Template image must be provided")
    
    # Make sure the frame and template are the same size
    if frame.shape != template_img.shape:
        template_img = cv2.resize(template_img, (frame.shape[1], frame.shape[0]))
    
    height = frame.shape[0]
    bottom_third_start = int(2 * height / 3)  # Start at 2/3 down the frame
    bottom_frame = frame[bottom_third_start:, :]
    bottom_template = template_img[bottom_third_start:, :]

    # Convert both images to grayscale for comparison
    frame_gray = cv2.cvtColor(bottom_frame, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(bottom_template, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Template Matching
    result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    
    # Method 2: Feature Matching
    # Initialize SIFT detector
    try:
        sift = cv2.SIFT_create()
        
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(frame_gray, None)
        kp2, des2 = sift.detectAndCompute(template_gray, None)
        
        # Calculate feature similarity
        feature_score = 0
        if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
            # Use FLANN matcher
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            # Get matches
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            # Calculate feature matching score
            feature_score = len(good_matches) / min(len(kp1), len(kp2)) if min(len(kp1), len(kp2)) > 0 else 0
    except Exception as e:
        # print(f"Warning: SIFT feature detection failed: {e}")
        feature_score = 0
    
    # Method 3: Mean-Squared Error (MSE) - simple alternative to SSIM
    # Calculate MSE between the two images
    try:
        from skimage.metrics import structural_similarity as ssim
        ssim_score = ssim(frame_gray, template_gray)
    except ImportError:
        # Fallback to MSE if skimage is not available
        mse = np.mean((frame_gray.astype("float") - template_gray.astype("float")) ** 2)
        # Convert MSE to a similarity score (inverse relationship)
        ssim_score = 1 / (1 + mse / 10000)  # Normalized to approximately 0-1 range
    
    # Combine scores (weighted average)
    combined_score = (0.5 * max_val) + (0.3 * feature_score) + (0.2 * ssim_score)
    
    # Print the score for debugging
    # print(f"Similarity score: {combined_score:.4f} (threshold: {similarity_threshold})")
    
    # Return True if the combined score exceeds the threshold
    return combined_score > similarity_threshold, combined_score

########################################################################################################################################################
########################################################################################################################################################

def pose_processing_worker(input_video_path, pitch_type, filename, template_img, x_percent, y_percent, width_percent, height_percent):
    """
    Function for extracting MediaPipe pose estimation data from a provided video clip.
    
    Parameters:
    input_video_path (str): Path to the input video derived from process_videos_to_json_parallel
    pitch_type (str): Type of pitch in the video derived from process_videos_to_json_parallel
    filename (str): Name of the video file derived from process_videos_to_json_parallel
    template_img (numpy.ndarray): Template image for comparison
    
    Returns:
    video_data (dict): Dictionary containing all landmark coordinates per frame 
    """
    # Initialize mp packages
    mp_pose = mp.solutions.pose
    
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    
    # Read the first frame to check for pitcher
    success, first_frame = cap.read()
    
    # Check if a pitcher is detected in the first frame using image similarity
    similarity_score = 0.0
    
    # Looks at bottom third of images which should be relatively consistent across samples showing just part of the pitcher 
    # and the field which should not change much 
    if template_img is not None:
        pitcher_detected, similarity_score = detect_pitcher(first_frame, template_img)
        
        # If no pitcher is detected, return early without further processing
        if not pitcher_detected:
            # print(f"Skipping {filename}: No pitcher detected (similarity score: {similarity_score:.4f})")
            return {
                'filename': filename,
                'pitch_type': pitch_type,
                'error': 'No pitcher detected',
                'landmark_data': {}
            }
    
    # Reset video capture to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Pose detector confidence threshold initializations. Thresholds don't really matter because the confidence score will be used as a parameter 
    with mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False) as pose:
        frame_count = 0
        
        # Landmark initialization 
        landmark_data = {i: [] for i in range(33)}
        
        # Loop through each frame in the video, limited to 300 frames
        while cap.isOpened() and frame_count < 300:
            success, frame = cap.read()
            if not success:
                break

            # Apply cropping
            frame_height, frame_width = frame.shape[:2]

            crop_x = int(frame_width * x_percent)
            crop_y = int(frame_height * y_percent)
            crop_width = int(frame_width * width_percent)
            crop_height = int(frame_height * height_percent)

            cropped_frame = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

            # Converting frame color to cv compatible color 
            rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            
            # Processing frame 
            pose_results = pose.process(rgb_frame)
            
            if pose_results.pose_landmarks:
                # Stores [xyz] coordinates per landmark per frame and visibility of the landmark 
                for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    landmark_data[idx].append([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
            frame_count += 1
    
    cap.release()
    
    # Create video data dictionary
    video_data = {
        'filename': filename,
        'pitch_type': pitch_type,
        'landmark_data': landmark_data
    }
    
    return video_data

########################################################################################################################################################
########################################################################################################################################################

def process_videos_to_json(base_directory, output_json_path, template_video_path, x_percent, y_percent, width_percent, height_percent):
    """
    Process videos in parallel using multiple CPU cores and save to JSON.
    
    Parameters:
    base_directory (str): Base directory containing subdirectories with videos
    output_json_path (str): Path to save the JSON output
    template_video_path (str, optional): Path to video whose first frame will be used as template
    max_frames (int): Maximum number of frames to process per video
    
    Returns:
    str: Path to the output JSON file
    """
    # Extract template image if path is provided
    template_img = extract_template_from_video(template_video_path)
    
    # Create a dictionary to hold all data
    all_data = {
        'metadata': {
            'landmark_names': [
                "nose", "left_eye_inner", "left_eye", "left_eye_outer", 
                "right_eye_inner", "right_eye", "right_eye_outer", 
                "left_ear", "right_ear", "mouth_left", "mouth_right", 
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
                "left_wrist", "right_wrist", "left_pinky", "right_pinky", 
                "left_index", "right_index", "left_thumb", "right_thumb", 
                "left_hip", "right_hip", "left_knee", "right_knee", 
                "left_ankle", "right_ankle", "left_heel", "right_heel", 
                "left_foot_index", "right_foot_index"
            ],
            'pitch_types': [],
        },
        'videos': []
    }
    
    # Track unique pitch types
    pitch_types = set()
    
    # Collect all videos to process
    videos_to_process = []
    
    # First pass to collect all videos
    for root, dirs, files in os.walk(base_directory):
        if root == base_directory:
            continue
            
        folder_name = os.path.basename(root)
        pitch_type = folder_name
        
        # Add this pitch type to set if it's new
        pitch_types.add(pitch_type)
        
        # Get all video files in the current directory
        video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mov'))]
        
        if not video_files:
            continue
            
        print(f"Found {len(video_files)} videos in folder: {folder_name} (Pitch type: {pitch_type})")
        
        # Add videos to the processing list
        for video_file in video_files:
            videos_to_process.append({
                'path': os.path.join(root, video_file),
                'filename': video_file,
                'pitch_type': pitch_type
            })
    
    # Determine optimal number of workers
    total_cores = os.cpu_count()
    num_workers = max(1, total_cores - 4) 
    
    # Process videos in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                pose_processing_worker, 
                video['path'], 
                video['pitch_type'],
                video['filename'],
                template_img,
                x_percent,
                y_percent,
                width_percent,
                height_percent
            ): video for video in videos_to_process
        }
        
        with tqdm.tqdm(total=len(videos_to_process), desc="Processing Videos") as pbar:
            for future in as_completed(futures):
                video_data = future.result()
                
                # Only include videos that were successfully processed
                if not video_data.get('error') or video_data.get('error') not in ['No pitcher detected']:
                    all_data['videos'].append(video_data)
                
                # Update progress
                pbar.update(1)
    
    # Update metadata
    all_data['metadata']['pitch_types'] = list(pitch_types)
    all_data['metadata']['total_videos'] = len(all_data['videos'])
    
    # Save to JSON file
    with open(output_json_path, 'w') as f:
        json.dump(all_data, f)
    
    print(f"All videos processed. Total: {len(all_data['videos'])} videos across {len(pitch_types)} pitch types.")
    # print(f"Data saved to {output_json_path}")
