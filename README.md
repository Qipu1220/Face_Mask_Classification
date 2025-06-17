
# 🎭 Face Mask Classification

**Real-time face mask detection using HOG/LBP features and RF/SVM classifiers**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-green)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Feature Illustration](#feature-illustration)
- [Classification Report](#classification-report)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

---

## 📌 Overview

This project was developed as part of the **CS231** course at **VNUHCM – University of Information Technology (UIT)**. It focuses on real-time detection of whether a person is wearing a face mask using a combination of:

- 🧠 Feature Extraction: **HOG** (Histogram of Oriented Gradients) and **LBP** (Local Binary Patterns)
- 🤖 Classifiers: **Random Forest** and **SVM**
- 👤 Face Detection: **YOLOv8** (via [Hugging Face Hub](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection))

---

## 🎬 Demo

![Demo](images/demo_camera.gif)

---

## ⚙️ Installation

### Prerequisites

- Python >= 3.8
- Recommended: use a virtual environment

```bash
python -m venv venv
source venv/bin/activate    # On Linux/macOS
venv\Scripts\activate       # On Windows
```

### Setup

```bash
git clone https://github.com/your-username/Face-Mask-Classification.git
cd Face-Mask-Classification
pip install -r requirements.txt
```

---

## 🚀 Usage

### Load Pre-trained Model

```python
from joblib import load

model = load('models/HOG_SVM_8x2.joblib')
# predictions = model.predict(feature_array)
```

See `example/load-model.ipynb` for instructions on how to build `feature_array`.

---

### Real-Time Detection with YOLOv8-Face

```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

repo_id = "arnabdhar/YOLOv8-Face-Detection"
model_path = hf_hub_download(repo_id=repo_id, filename="model.pt")

face_detector = YOLO(model_path)
results = face_detector(frame)
# Extract face → compute features → classify
```

---

### Run Real-Time Demo

```bash
python camera_integration/demo_model_Withdetec.py
```

---

## 🗂️ Project Structure

[Project tree omitted for brevity...]

---

## 📁 Dataset

We use the [Face Mask 12K Images Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)

---

## 📊 Feature Illustration

- HOG Features: `images/hog_8x2.png`
- LBP Features: `images/LBP.png`

---

## 📈 Classification Report

| Model                 | Precision | Recall | F1-Score | Accuracy |
|-----------------------|:---------:|:------:|:--------:|:--------:|
| HOG + Random Forest   | 0.98      | 0.98   | 0.98     | 0.98     |
| HOG + SVM             | 0.99      | 0.99   | 0.99     | 0.99     |
| LBP + Random Forest   | 0.97      | 0.97   | 0.97     | 0.97     |
| LBP + SVM             | 0.97      | 0.97   | 0.97     | 0.97     |

---

## 🤝 Contributing

We welcome all contributions! To contribute:

1. Fork the repository  
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)  
3. Commit your changes  
4. Push to the branch  
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

- **Course**: CS231 – University of Information Technology (UIT), VNU-HCM  
- **Team**: UIT-ChickenPlusPlus  
- **Email**: 23521672@gm.uit.edu.vn

---

## 🙏 Acknowledgements

- Kaggle: Face Mask 12K Images Dataset  
- Hugging Face Hub: YOLOv8 Face Detection  
- UIT – CS231 Course
