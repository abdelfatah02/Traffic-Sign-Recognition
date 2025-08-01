# Traffic-Sign-Recognition

## 📌 Overview

This project implements a **multi-class image classification model** to recognize and classify **German traffic signs** using the **GTSRB dataset**. It leverages **Convolutional Neural Networks (CNNs)** to accurately identify traffic sign categories based on their visual features.

---

## 🗂️ Dataset

- **Source:** [GTSRB - German Traffic Sign Recognition Benchmark (Kaggle)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Records:** 39,000+ images for training, 12,000+ for testing
- **Format:** Color images (varied sizes)
- **Classes:** 43 traffic sign classes, labeled from `0` to `42`

---

## 🎯 Objectives

- Load and preprocess traffic sign images (resize, normalize)
- Encode class labels for multi-class learning
- Train a CNN to classify traffic signs
- Evaluate model performance using accuracy and confusion matrix
- Visualize learning progress and predictions

---

## 🛠️ Tools & Libraries

- **Python**
- **TensorFlow / Keras** – CNN model building and training
- **OpenCV** – image reading and processing
- **NumPy / Pandas** – data manipulation
- **Matplotlib / Seaborn** – visualization and plotting

---

## 📊 Key Components

### 1. 🖼️ Image Preprocessing

- Loaded and resized all images to a fixed shape `(32, 32, 3)`
- Normalized pixel values to range `[0, 1]` for faster training
- Encoded labels using one-hot encoding

### 2. 🔁 Train/Test Split

- Used `train_test_split()` with stratification to ensure class balance
- Split into `training` and `validation` sets

### 3. 🧠 CNN Model

- Built a **custom CNN architecture** with:
  - Multiple convolutional and pooling layers
  - Dropout for regularization
  - Dense output layer with 43 neurons and softmax activation

### 4. 🚂 Training

- Trained model using:
  - Optimizer: `Adam`
  - Loss Function: `categorical_crossentropy`
  - Metrics: `accuracy`
- Used validation set to monitor overfitting

### 5. 📈 Evaluation

- Visualized training vs. validation **accuracy** and **loss** across epochs
- Evaluated the model on **unseen test images**
- Computed and plotted a **confusion matrix** to inspect class-level performance

---

## 🧠 Results

- Achieved high accuracy on the validation set
- Most classes classified correctly with minimal confusion
- CNN was able to generalize well on real traffic signs

---

## ✅ Covered Topics

- Deep Learning with CNNs
- Computer Vision & Image Preprocessing
- Multi-class Classification
- Model Evaluation (Accuracy, Loss, Confusion Matrix)
- TensorFlow/Keras Practice

---

## 🏁 Future Improvements

- Try **pre-trained models** like MobileNet or ResNet (transfer learning)
- Apply **data augmentation** to improve generalization
- Convert model to **TFLite** for mobile deployment
- Integrate with a **real-time video feed** to recognize signs live

