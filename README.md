# Yoga Pose Estimation and Classification using YOLOv8

## Project Overview

This project focuses on human pose estimation using the YOLOv8 model, particularly for yoga poses. With the increasing popularity of yoga and the need for technology-assisted learning platforms, this project aims to provide accurate pose detection to ensure safe and effective yoga practice. The project specifically detects key points corresponding to different yoga poses and classifies them accordingly.

## Key Features

- **YOLOv8 Model**: Utilizes the YOLOv8 architecture, known for its efficiency in detecting key points in images.
- **Yoga Pose Classification**: Classifies five distinct yoga poses: Downdog, Goddess, Plank, Tree, and Warrior2.
- **Dataset**: The dataset used consists of yoga pose images and is divided into train and test sets.
- **Accuracy**: The YOLOv8 model achieved an accuracy of 88% in classifying yoga poses.

## Dataset

- **Source**: The dataset is publicly available on Kaggle.
- **Structure**: The dataset is divided into two folders: Train (1081 images) and Test (470 images).
- **Classes**: The dataset includes five yoga poses: Downdog, Goddess, Plank, Tree, and Warrior2.

## Model Architecture

The YOLOv8 model is used to detect 17 key points on the human body corresponding to different yoga poses. The key points detected include joints such as shoulders, elbows, wrists, hips, knees, and ankles. After detecting the key points, a custom neural network classifier is used to map these key points to the corresponding yoga pose labels.

### Preprocessing

- Images are resized, and their color channels are converted from BGR to RGB.
- Key points detected are stored in a CSV file, which includes the image name, class label, and key point coordinates.

### Classifier

The YOLOv8-based classifier is built with multiple convolutional layers followed by fully connected layers. The ReLU activation function is used in hidden layers, and the Softmax activation function is applied in the output layer for multi-class classification.

### Optimization

The model is optimized using the Adam optimizer, which dynamically adjusts the learning rate for faster convergence and improved performance.

## Results

- **Accuracy**: The YOLOv8 model achieved an accuracy of 88% in classifying yoga poses.
- **Evaluation Metrics**: The model was evaluated using precision, recall, and F1 score.

## Future Scope

1. Future improvements could focus on enhancing the YOLOv8 model's accuracy by incorporating advanced data augmentation techniques, such as rotation, scaling, and noise injection, to better generalize across diverse yoga poses and environments. 
2. Additionally, integrating real-time feedback systems to provide users with immediate corrections for improper form can make the model more practical in real-world applications. 
3. Exploring transfer learning and fine-tuning with larger, more varied datasets can further improve performance. 
4. Implementing pose correction algorithms using the detected key points will enable more accurate posture assessments.
