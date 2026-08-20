# Waste Classification

An AI-powered waste classification project that uses **Deep Learning and Transfer Learning** to classify waste images into different categories.

The project uses a **fine-tuned EfficientNetB0** model and provides a **Streamlit web application** where users can upload an image and get the predicted waste category along with the model's confidence.

---

## Features

- Image-based waste classification
- 7 different waste categories
- Transfer learning using EfficientNetB0
- Fine-tuning of pretrained layers
- Data augmentation
- Early stopping during training
- Classification report and model evaluation
- Interactive Streamlit web application
- Confidence score for predictions

---

## Waste Categories

The model classifies images into:

- Battery
- Cardboard
- Clothes
- Glass
- Metal
- Paper
- Plastic

---

## Model

The project uses **EfficientNetB0**, a pretrained convolutional neural network originally trained on the ImageNet dataset.

### Transfer Learning Approach

The model was trained in two stages:

1. **Feature Extraction**
   - EfficientNetB0 was initialized with ImageNet pretrained weights.
   - The pretrained layers were initially frozen.
   - A custom classification head was added for the 7 waste categories.

2. **Fine-Tuning**
   - Selected pretrained layers were unfrozen.
   - The model was trained using a small learning rate.
   - This allowed EfficientNetB0 to adapt its visual features specifically to waste images.

### Architecture

````text
Input Image
     ↓
Data Augmentation
     ↓
EfficientNetB0
(Pretrained on ImageNet)
     ↓
Fine-Tuned Layers
     ↓
Global Average Pooling
     ↓
Dense Layer (128)
     ↓
Dropout
     ↓
Dense Layer (7)
     ↓
Softmax
     ↓
Waste Category

````
---

## Model Performance

The final fine-tuned EfficientNetB0 achieved approximately:

### 90% Validation Accuracy

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Battery | 0.96 | 0.96 | 0.96 |
| Cardboard | 0.98 | 0.86 | 0.91 |
| Clothes | 0.94 | 0.99 | 0.96 |
| Glass | 0.88 | 0.88 | 0.88 |
| Metal | 0.85 | 0.92 | 0.88 |
| Paper | 0.89 | 0.91 | 0.90 |
| Plastic | 0.90 | 0.78 | 0.83 |

### Overall Metrics

- **Accuracy:** 90%
- **Macro F1-Score:** 0.90
- **Weighted F1-Score:** 0.90

---

## Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB0
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Pillow

---

## Project Structure

```text
waste-classifier/
│
├── waste_dataset.zip
├── app.py
├── waste_classifier.keras
├── class_names.json
├── requirements.txt
├── README.md
└── waste_classification.ipynb
````

### File Description

| File                         | Description                            |
| ---------------------------- | -------------------------------------- |
| `app.py`                     | Streamlit web application              |
| `waste_classifier.keras`     | Trained EfficientNetB0 model           |
| `class_names.json`           | Waste category labels                  |
| `requirements.txt`           | Python dependencies                    |
| `waste_classification.ipynb` | Model training and evaluation notebook |
| `README.md`                  | Project documentation                  |
| `waste_dataset.zip`          | Waste image dataset                    |

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd waste-classifier
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## How to Use

1. Open the Streamlit application.
2. Upload an image of a waste item.
3. Click **Predict Waste Type**.
4. The model analyzes the image.
5. The predicted waste category and confidence score are displayed.

### Example

```text
Prediction

Paper

Confidence: 92.34%
```

---

## Model Training

The model training process included:

```text
Dataset
   ↓
Image Preprocessing
   ↓
Data Augmentation
   ↓
EfficientNetB0
   ↓
Transfer Learning
   ↓
Fine-Tuning
   ↓
Early Stopping
   ↓
Model Evaluation
```

The dataset was divided into training and validation data for model development and evaluation.

---

## 🔮 Future Improvements

- Add more waste categories
- Increase the size and diversity of the dataset
- Improve classification of visually similar materials
- Add real-time camera-based classification
- Deploy the application online
- Add recycling and disposal recommendations
- Improve performance on difficult classes such as Plastic
- Add confidence visualization for all seven categories

---

## 👩‍💻 Author

**Riddhima Mehra**

B.Tech – Artificial Intelligence & Machine Learning
