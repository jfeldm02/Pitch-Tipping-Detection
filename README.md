# Pitch-Tipping Detection via Human-Pose Estimation & Time-Series Classification  
**Author:** *Justin Feldman*  
**Course:** CS6140 – Machine Learning, Northeastern University (Spring 2025)

---

## Overview  
This project proposes a methodology for **classifying baseball pitch types** using biomechanical data extracted from video samples via computer vision pose-estimation. Specifically, it identifies **subtle, repetitive body movements** (a phenomenon known as *pitch-tipping*) to predict pitch types before release.

By combining **pose estimation**, **time-series modeling**, and **explainable ML techniques** such as **t-SNE** and **UMAP**, this study provides both predictive models and interpretability tools to aid hitters in exploiting tells and helping pitchers eliminate them.

![image](https://github.com/user-attachments/assets/bdbce777-397a-4894-b55a-4d2c198223f2)

---

## Repository Structure

| File/Directory                | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `pose_estimation.py`         | Mediapipe-based pose estimation script (called by `main_data_acquisition.py`) |
| `data_preprocessing.py`      | Prepares raw pose data for modeling and analysis (called by `main_data_acquisition.py`)                          |
| `main_data_acquisition.py`   | End-to-end data pipeline: loads videos, extracts pose data, and preprocesses |
| `video_cropping_calibration.ipynb` | Cropping calibrator for improving pose estimation quality                  |
| `pitcher_modeling.ipynb`     | Builds and evaluates pitch classification models                          |
| `data_exploration.ipynb`     | Explains model predictions through visualizations and feature analysis     |
| `dependencies.txt`           | Required Python packages and versions                                      |
| `pitcher_sample_datasets/`   | Sample biomechanical data for experimentation                             |
| `Fastball.zip`          | A few sample Brayan Bello fastball videos                |
| `Changeup.zip`          | A few sample Brayan Bello changeup videos                |

---

## Problem Statement  
**Pitch-tipping** refers to small, consistent variations in a pitcher’s motion that may give away their next pitch. This project leverages computer vision and machine learning to:  
- Detect those subtle cues in pose data  
- Accurately classify pitch types  
- Explain predictions in an interpretable way for practical use

---

## Approach

### Dataset  
Video samples are sourced from the **MLB Film Room**, a comprehensive and labeled archive of Major League Baseball broadcast footage since 2017.

### Pose Estimation  
Pose keypoints are extracted from each frame using **Google’s Mediapipe**, which uses:
- **CNNs** to detect regions of interest
- A secondary **landmark regression model** to estimate human joint positions

### Preprocessing Steps  
- **Void filtering:** Discards low-quality samples characterized by poor camera angles or footage interruptions  
- **Feature consolidation:** Combines related landmarks to reduce feature count and improve data smoothness  
- **Temporal alignment:** Normalizes to a 1.7s (100 frames/60 FPS) window before pitch release  
- **Downsampling:** Uses every 4th frame for efficiency and data smoothness  
- **Scaling:** Converts pixel values to inches using pitcher height  
- **Normalization:** Centers data on a consistent body landmark reference point  

### Model Architectures  
Three main approaches were tested:
- **Multinomial Logistic Regression**
- **Support Vector Machines (SVM)**
- **Deep Neural Networks (DNN)**
  - Recurrent Neural Network (RNN)
  - Long Short-Term Memory (LSTM)
  - Convolutional Neural Network (CNN)

A **genetic algorithm** was used for automated hyperparameter tuning of RNN/LSTM architectures.

### Training Pipeline  
- Input: 25-frame sequence, equally spaced over 100 frames of footage, of pose landmarks  
- Features: x, y, and confidence per landmark per frame  
- Models: Trained on stratified train/test splits  
- Optimizer: Adam with cross-entropy loss  

### Evaluation Methods  
- Landmark plots (per-pitch and averaged)  
- t-SNE / UMAP visualizations per landmark  
- Accuracy scores per model and class  
These techniques offer visual, interpretable insight into pitch-tipping indicators.

---

## Results  
*[Add training/test accuracy plots here]*  
*[Add sample UMAP/t-SNE plots highlighting key indicators]*  

---

## Setup & Installation  

### Dependencies  
Install all required packages in a terminal using:

```bash
pip install -r dependencies.txt
```
### Running the Code

1. **Download all files** and place them in your directory of choice.

2. **Use a Sample Processed Dataset** or **Calibrate Cropping Window**  
   - Select a .json file from `pitcher_sample_datasets/` and skip to **Step 4** or...
   - Select a good sample video from either your own dataset or the provided sample dataset.  
   - This video should start with a standard pitching angle, as shown below.  
   - Open `video_cropping_calibration.ipynb`, provide the video path, and adjust the cropping variables to find the optimal range, also as shown below.  
   
   ![image](https://github.com/user-attachments/assets/a023845f-c1c4-4a57-babf-3ea0498ddd7a)

3. **Configure and Run Data Pipeline**  
   - Separate your video subdirectories into their own standalone directory. The script will read all directories present and present an error if other folders are present.  
   - Open `main_data_acquisition.py` and fill in the following variables:

     ```python
     pitcher_height = ...  # Height of the pitcher in inches
     pitcher_handedness_right = True  # or False if left-handed
     main_directory = 'path/to/video/folders/'
     raw_data_output = 'path/to/save/raw/'
     processed_data_output = 'path/to/save/processed/'
     good_video_sample = 'path/to/sample/video.mp4'
     x_percent, y_percent, width_percent, height_percent = ...  # From calibration notebook
     ```

   - Once filled out, run the script. It will:
     - Parse through all labeled video directories  
     - Filter acceptable samples  
     - Run pose estimation  
     - Save both raw and processed datasets to the specified paths  

   - For reference: The full sample dataset (~1200 videos) takes ~45 minutes using 8 CPU cores.

4. **Train and Explore Models**  
   - Open both `pitcher_modeling.ipynb` and `data_exploration.ipynb`.  
   - Fill in the following:

     ```python
     pitcher_name = 'First Last'  # For labeling plots
     processed_data_path = 'path/to/processed.csv'
     save_your_models = 'path/to/save/models/'
     ```

   - Run all cells.  
     - Observe model train/test accuracy to determine how likely pitch-tipping is.  
     - High test accuracy may indicate consistent, detectable pre-pitch patterns.  

5. **Interpret Results**  
   - Use methods in `data_exploration.ipynb` to:
     - Identify which **body landmarks** show the strongest tipping signal  
     - Visualize pose-based clusters using **t-SNE** and **UMAP**  
     - Review results alongside video to make **actionable mechanical insights**

---

### Future Work

This project’s focus was on **model explainability**, aiming to uncover **why** a model predicts a certain pitch. Current tools like **t-SNE**, **UMAP**, and landmark comparisons offer insight, but they don’t *automatically* explain model behavior.

#### Other Future Endeavors:
- Deeper integration of **SHAP** for individualized prediction breakdowns  
- Visual **animated landmark sequences** to intuitively show pitch-tipping behavior  
- Apply this framework to new areas like:
  - Injury/fatigue indicators  
  - Performance slump prediction  
  - Pitch velocity estimation   

---

### References

- **Cao et al., 2019** — *OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields.* IEEE Transactions on PAMI  
- **Fawaz et al., 2019** — *Deep learning for time series classification: a review.* Data Mining and Knowledge Discovery, 33(4), Springer  

---

### More Information

This project was completed in **Spring 2025** for **CS6140: Machine Learning** during my **Master’s in Artificial Intelligence** at **Northeastern University**.

**Portfolio**: https://www.justinafeldman.com/
