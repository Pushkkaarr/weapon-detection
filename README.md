# 🛡️ Real-Time Weapon Detection System for Security Surveillance

This project is a computer vision system designed to detect weapons in video streams or files. It is intended to enhance public safety by identifying dangerous objects such as firearms in real-time or pre-recorded surveillance footage.

---

## 🎯 Target Weapons for Detection

The system can detect the following:

- **Primary Weapon Classes (Required)**
  - 🔫 **Firearms**: handguns, pistols, rifles.
  - ✅ **No Weapon**: normal or safe scenarios.


> **Note:** The system is trained to minimize false alarms from harmless objects, such as toys

---

## 🔧 Project Modules

### 1️⃣ `video_inference.py`
- **Purpose:** Processes a video and produces an output video with red bounding boxes around detected weapons.
- **How it works:**
  - Loads the trained Faster R-CNN model (`detector_epochX.pth`).
  - Reads a video file frame by frame.
  - Runs object detection on each frame.
  - Draws bounding boxes and class labels on detected weapons.
  - Saves the resulting video as `.avi` or `.mp4`.
- **Configurable paths:**
  ```python
  input_video_path = "input_video.mp4"  # Set your original video here
  output_video_path = "output.avi"      # Video with weapon detections
  checkpoint_path = "models/detector_epoch3.pth"  # Trained model path
* **Run Command:**

  ```bash
  python src/video_inference.py
  ```



### 2️⃣ `extract_weapon_frames.py`

* **Purpose:** Extracts frames from a video where weapons are detected, and saves them as images with bounding boxes applied.
* **How it works:**

  * Loads the trained Faster R-CNN model.
  * Reads the input video frame by frame.
  * For frames with weapons detected above a confidence threshold, saves the frame to a folder.
  * Names the frames with the timestamp in seconds where the weapon appears.
* **Configurable paths:**

  ```python
  input_video_path = "input_video.mp4"   # Original video path
  frames_output_dir = "weapon_frames/"   # Folder to save extracted frames
  checkpoint_path = "models/detector_epoch3.pth"  # Trained model
  ```
* **Run Command:**

  ```bash
  python src/extract_weapon_frames.py
  ```
* **Output:** Each frame image contains red boxes around weapons and is named as `frame_XXs.jpg`.

---

### 3️⃣ `train_detector.py`

* **Purpose:** Train the Faster R-CNN model on your annotated dataset.
* **How it works:**

  * Reads images and JSON annotations from `data/train` and `data/val`.
  * Creates a PyTorch dataset using `JsonDetectionDataset`.
  * Trains a Faster-RCNN model.
  * Saves checkpoints periodically and after each epoch.
* **Configurable paths & params:**

  ```bash
  python src/train_detector.py --data-dir data --epochs 5 --batch-size 2 --device cuda
  ```
* **Resume Training from Checkpoint:**

  ```bash
  python src/train_detector.py --data-dir data --epochs 5 --batch-size 2 --device cuda --resume models/checkpoint_epoch2_batch1000.pth
  ```

---


## ⚙️ Installation & Setup

### **Installation**

```bash
# Clone the repository
git clone https://github.com/Pushkkaarr/weapon-detection-system.git

# Navigate to the project folder
cd weapon-detection

# Install dependencies
pip install -r requirements.txt
```

---

## 🗂️ Directory Structure

```
weapon-detection/
│
├─ src/
│   ├─ train_detector.py          # Training script
│   ├─ video_inference.py        # Weapon detection on videos
│   ├─ extract_weapon_frames.py  # Extract frames with weapons
│   ├─ evaluate_model.py         # Evaluate accuracy
│   └─ dataset.py                # JsonDetectionDataset class
│
├─ data/
│   ├─ train/images/             # Training images
│   ├─ train/labels/             # JSON annotations
│   ├─ val/images/               # Validation images
│   └─ val/labels/               # JSON annotations
│
├─ models/                       # Saved model checkpoints
├─ weapon_frames/                # Extracted frames with weapons
└─ README.md
```

---

## 📝 How to Update Paths

* **Videos:** Change `input_video_path` in `video_inference.py` and `extract_weapon_frames.py` to the video you want to process.
* **Output folder:** Change `output_video_path` or `frames_output_dir` in the scripts.
* **Trained Model:** Ensure the correct checkpoint is loaded (`detector_epochX.pth`).

---

## 🚀 Running the System

1. **Train the model (optional if already trained):**

   ```bash
   python src/train_detector.py --data-dir data --epochs 5 --batch-size 2 --device cuda
   ```

2. **Evaluate model accuracy:**

   ```bash
   python src/evaluate_model.py
   ```

3. **Detect weapons in video:**

   ```bash
   python src/video_inference.py
   ```

4. **Extract frames with detected weapons:**

   ```bash
   python src/extract_weapon_frames.py
   ```

---

## 📂 Output

* **Video Showing the video processing and actual execution of model:**
(Video is compressed , hence low quality)

https://github.com/user-attachments/assets/95e69b89-df52-4771-8398-c9480013dfc0



* **Weapon Detection Video:** Saved at `output.avi` or your configured path in `video_inference.py`.
* 

https://github.com/user-attachments/assets/1b8fa097-8a50-4812-a261-809f0ff177b0


* **Weapon Frames:** Saved in `weapon_frames/` with names like `frame_12s.jpg` for frame at 12 seconds where a weapon is detected.
* <img width="1688" height="833" alt="image" src="https://github.com/user-attachments/assets/b0ee3316-799f-4e18-b0a8-d1f550edcb95" />


