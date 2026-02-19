# 🥔 Potato Leaf Disease Detection Using Deep Learning (CNN)

A deep learning-based automated system for detecting and classifying potato leaf diseases using a custom Convolutional Neural Network (CNN).  
This project identifies **Early Blight**, **Late Blight**, and **Healthy** potato leaves with high accuracy to support precision agriculture.

---

## 📄 Research Paper

**Title:** Advancing Potato Crop Health and Disease Detection with Deep Learning
**Authors:** Ayush Kumar, Naveen Kumar Tiwari, Abhishek Bajpai, BDK Patro  


📊 Achieved **97.4% average accuracy** using 5-Fold Cross Validation.

🔗 **DOI (Official Publication):** https://doi.org/10.1007/978-981-96-9083-1_12
---


## 🚀 Features

- Custom 20-layer CNN architecture  
- Multi-class classification (3 leaf conditions)  
- Image preprocessing & augmentation  
- K-Fold Cross Validation  
- High performance metrics (Accuracy, Precision, Recall, F1-score)  
- ROC curves and confusion matrices  

---

## 🧠 Classes Detected

| Class | Description |
|------|------------|
| Healthy | Normal potato leaf |
| Early Blight | Fungal leaf disease |
| Late Blight | Severe blight infection |

## Sample Image
<p align="center">
  <img src="Advancing-Potato-Crop-Health-and-Disease-Detection-using-Deep-Learning/Image Of Code/Sample Image.png" width="750">
</p>
---

## 📂 Dataset

**PlantVillage Dataset**

### Preprocessing Steps:
- Resize to 256×256  
- Normalize pixel values (0–1)  
- Data augmentation:
  - Rotation  
  - Flipping  
  - Cropping  

---

## 🏗 Model Architecture

- 5 Convolutional Blocks (16 → 32 → 64 → 128 → 256 filters)  
- ReLU activation  
- MaxPooling layers  
- Fully connected dense layers  
- Softmax output for classification  


<p align="center">
  <img src="Advancing-Potato-Crop-Health-and-Disease-Detection-using-Deep-Learning/Image Of Code/Work Chart.png" width="750">
</p>
---

## 📈 Performance

| Metric | Value |
|-------|------|
| Average Accuracy | 97.4% |
| Best Fold Accuracy | 99.9% |
| Precision | 0.95 |
| Recall | 0.97 |
| F1-Score | 0.96 |

<p align="center">
  <img src="Advancing-Potato-Crop-Health-and-Disease-Detection-using-Deep-Learning/Image Of Code/FOLD 1/Loss.png" width="420" style="border:2px solid #ccc"/>
  <img src="Advancing-Potato-Crop-Health-and-Disease-Detection-using-Deep-Learning/Image Of Code/FOLD 1/Accuracy.png" width="420" style="border:2px solid #ccc"/><br>
  <img src="Advancing-Potato-Crop-Health-and-Disease-Detection-using-Deep-Learning/Image Of Code/FOLD 1/Confusion Matrix 1.png" width="420" style="border:2px solid #ccc"/>
  <img src="Advancing-Potato-Crop-Health-and-Disease-Detection-using-Deep-Learning/Image Of Code/FOLD 1/ROC Curve 1.png" width="420" style="border:2px solid #ccc"/>
</p>



## Prediction Images
<p align="center">
<img src="Advancing-Potato-Crop-Health-and-Disease-Detection-using-Deep-Learning/Image Of Code/Prediction Images.png" width="800"/>
  </p>
### Comparison:

| Model | Accuracy |
|------|---------|
| MobileNetV2 | 86.11% |
| VGG16 | 95% |
| CNN-Transformer | 95% |
| EfficientRMT-Net | 96% |
| Proposed CNN | 97.4% |

---

## 🛠 Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- OpenCV  
- Scikit-learn  
- Matplotlib  

---

