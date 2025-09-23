# 🛡️ Real-Time Weapon Detection System for Security Surveillance

This project is a computer vision system designed to enhance public safety by detecting dangerous weapons in real-time surveillance footage. By leveraging deep learning, the system identifies potential security threats, allowing for proactive intervention before incidents occur.

-----

## 🎯 Target Weapons for Detection

The model is trained to identify specific categories of weapons while minimizing false alarms from harmless objects.

#### **Primary Weapon Classes (Required)**

  * 🔫 **Firearms**: Includes handguns, pistols, and rifles.
  * ✅ **No Weapon**: Represents normal, safe scenarios for baseline comparison and reducing false positives.

#### **Advanced Categories (Bonus Points)**

  * 🔪 **Improvised Weapons**: Objects that can be used as weapons, such as broken bottles and metal rods.

> **Note:** A key challenge is to distinguish between actual threats and visually similar, harmless objects (e.g., a toy gun, a kitchen knife in a restaurant). The system is being developed to understand context and minimize false alarms.

-----

## 🔧 Core Requirements

The project is built upon three main modules that form a complete video analysis pipeline.

1.  **Data Processing Module**

      * Handles various video inputs (e.g., live streams, pre-recorded files).
      * Responsible for efficient frame extraction and pre-processing for the model.

2.  **Model Implementation**

      * Utilizes a deep learning model (e.g., YOLO, SSD, or a custom CNN) for robust object detection and classification.
      * The model is trained to accurately classify the target weapon classes.

3.  **Video Analysis Pipeline**

      * Integrates the data processing and modeling modules into a seamless pipeline.
      * Processes video files or streams and generates real-time threat detection results, including bounding boxes and threat labels.

-----

## 📅 Timeline & Submission

  * **Deadline**: September 21, 2025, 11:59 PM IST
  * **Submission Format**: A link to a public GitHub repository containing the complete source code, trained models, and detailed documentation.

-----

## 🚀 Getting Started

### **Prerequisites**

A list of software and libraries required to run the project.

  * Python 3.8+
  * OpenCV
  * TensorFlow / PyTorch
  * NumPy

### **Installation**

A step-by-step guide to get the development environment running.

```bash
# Clone the repository
git clone https://github.com/Pushkkaarr/weapon-detection-system.git

# Navigate to the project directory
cd weapon-detection

# Install the required packages
pip install -r requirements.txt
```

