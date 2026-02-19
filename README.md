# One-shot-face-recognition-using-Siamese-Neural-Networks


![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![IEEE](https://img.shields.io/badge/Published-IEEE%20INCET%202025-blue)

---

## 🏆 Research Publication

This work was published at:

**6th International Conference on Emerging Technology (INCET 2025)**  
IEEE Indexed Conference

The project evaluates three Siamese Network architectures for one-shot face recognition.

Final Test Accuracy: **94%**

---

## 🧠 Siamese Network Architecture

![Architecture](siamese%20neural%20network.png)


---

## 🏗️ Backbone Architectures

### 🔹 ResNet18 Based Siamese Network

![ResNet18](Siamese%20network%20using%20ResNet18%20as%20backbone.png)


### 🔹 MobileNetV3 + SE Based Siamese Network

![MobileNetV3](Siamese%20network%20using%20MobileNetV3%20as%20backbone.png)


---

## 📊 Training Results

### Accuracy Curve

![Accuracy](Accuracy%20results.png)

### Loss Curve

![Loss](Loss%20results.png)

---

## 📈 Model Comparison

![Comparison](model_comparison.png)

---

## 🛠️ Tech Stack

- Python
- PyTorch
- ResNet18
- MobileNetV3
- Transfer Learning
- Siamese Networks
- One-Shot Learning
- Computer Vision

---

## 🚀 Methodology

1. Images are organized by identity folders.
2. Anchor, Positive, and Negative samples are generated dynamically.
3. Feature embeddings are extracted using pretrained backbones.
4. Distance between embeddings is passed through a logistic layer.
5. BCEWithLogitsLoss is used for similarity classification.

---

## 📂 Dataset Structure

