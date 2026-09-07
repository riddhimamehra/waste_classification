# EcoSort: Automated Waste Classification using Transfer Learning

An AI-powered waste classification project that uses **Deep Learning and Transfer Learning** to classify waste images into 7 categories, benchmarking a custom CNN against MobileNetV2 and EfficientNetB0 transfer-learning backbones.

🔗 **Live demo:** [add your Streamlit Community Cloud URL here]

The project uses a **fine-tuned EfficientNetB0** model (compared against a MobileNetV2 baseline) and provides a **Streamlit web application** where users can upload an image and get the predicted waste category along with the model's confidence.

---

## Features

- Image-based waste classification across 7 categories
- Transfer learning using EfficientNetB0, benchmarked against MobileNetV2
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

The project compares two pretrained convolutional backbones, both originally trained on ImageNet: **EfficientNetB0** (the final deployed model) and **MobileNetV2** (used as a lighter-weight comparison point).

### Transfer Learning Approach

Both models were trained in two stages:

1. **Feature Extraction**
   - The backbone was initialized with ImageNet pretrained weights.
   - The pretrained layers were initially frozen.
   - A custom classification head was added for the 7 waste categories.

2. **Fine-Tuning**
   - Selected pretrained layers were unfrozen.
   - The model was trained using a small learning rate.
   - This allowed the backbone to adapt its visual features specifically to waste images.

### Architecture

```
Input Image
     ↓
Data Augmentation
     ↓
Backbone (EfficientNetB0 / MobileNetV2)
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
```

---

## Model Comparison: Custom CNN vs. MobileNetV2 vs. EfficientNetB0

Three architectures were trained and evaluated on the same dataset split: a custom CNN built from scratch (baseline), MobileNetV2 (transfer learning), and EfficientNetB0 (transfer learning, the deployed model).

| Model | Validation Accuracy | Macro F1-Score | Notes |
|---|---|---|---|
| Custom CNN (from scratch) | 70.46% | 0.71 | No pretrained weights — baseline |
| MobileNetV2 (feature extraction) | 61.62% | — | Backbone frozen |
| MobileNetV2 (fine-tuned) | 61.26% | 0.57 | Fine-tuning slightly *decreased* accuracy; near-total failure on the Plastic class (see below) |
| EfficientNetB0 (feature extraction) | 50.36% | — | Backbone frozen — weakest starting point |
| **EfficientNetB0 (fine-tuned, deployed)** | **90.31%** | **0.90** | Same fine-tuning procedure as MobileNetV2 |

**Takeaway:** The results weren't what a "bigger model wins" assumption would predict. MobileNetV2 underperformed even the from-scratch custom CNN, and fine-tuning barely changed its accuracy (a slight *decrease*, from 61.62% to 61.26%). EfficientNetB0 actually started from a weaker frozen-backbone accuracy (50.36% vs. MobileNetV2's 61.62%), but responded dramatically better to the identical fine-tuning procedure — unfreezing the last 30 layers and training with a low learning rate — jumping to 90.31%. This suggests EfficientNetB0's pretrained features were better suited to adapt to this waste-classification task once fine-tuned, while MobileNetV2's frozen features were closer to a local optimum that the same fine-tuning budget couldn't meaningfully improve on.

The per-class breakdown makes MobileNetV2's weakness concrete rather than just a lower aggregate number: its recall on **Plastic dropped to 0.05** — it correctly identified only 5% of actual Plastic images, essentially failing on that class entirely, while still holding reasonable precision (0.62) on the few it did catch. EfficientNetB0, by contrast, kept Plastic recall at 0.78 — still its weakest class, but nowhere near collapse. This points to Plastic being a genuinely hard category (visually variable, easily confused with Glass or Metal in some samples) that only survives with a backbone that adapts well during fine-tuning.

Overall, this is a useful reminder that transfer learning performance depends heavily on how well a specific backbone's fine-tuning responds to the target dataset, not just on the backbone's reputation or parameter count.

See `waste_classification.ipynb` for the full training and evaluation code for all three models.

---

## Model Performance (EfficientNetB0 — deployed model)

The final fine-tuned EfficientNetB0 achieved approximately:

### 90% Validation Accuracy

| Class     | Precision | Recall | F1-Score |
| --------- | --------- | ------ | -------- |
| Battery   | 0.96      | 0.96   | 0.96     |
| Cardboard | 0.98      | 0.86   | 0.91     |
| Clothes   | 0.94      | 0.99   | 0.96     |
| Glass     | 0.88      | 0.88   | 0.88     |
| Metal     | 0.85      | 0.92   | 0.88     |
| Paper     | 0.89      | 0.91   | 0.90     |
| Plastic   | 0.90      | 0.78   | 0.83     |

### Overall Metrics

- **Accuracy:** 90%
- **Macro F1-Score:** 0.90
- **Weighted F1-Score:** 0.90

---

## Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB0, MobileNetV2, custom CNN
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Pillow

---

## Project Structure

```
EcoSort/
│
├── waste_dataset.zip
├── app.py
├── waste_classifier.keras
├── class_names.json
├── requirements.txt
├── README.md
└── waste_classification.ipynb
```

### File Description

| File                         | Description                            |
| ---------------------------- | -------------------------------------- |
| `app.py`                     | Streamlit web application              |
| `waste_classifier.keras`     | Trained EfficientNetB0 model           |
| `class_names.json`           | Waste category labels                  |
| `requirements.txt`           | Python dependencies                    |
| `waste_classification.ipynb` | Model training, comparison, and evaluation notebook |
| `README.md`                  | Project documentation                  |
| `waste_dataset.zip`          | Waste image dataset                    |

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/riddhimamehra/waste_classification.git
cd waste_classification
```

### 2. Create a virtual environment

```
python -m venv venv
```

Activate it on Windows:

```
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## Run the Application

Start the Streamlit application using:

```
streamlit run app.py
```

The application will open in your browser.

---

## How to Use

1. Open the Streamlit application.
2. Upload an image of a waste item.
3. Click **Predict**.
4. The model analyzes the image.
5. The predicted waste category and confidence score are displayed.

### Example

```
Prediction

♻️ Paper

Confidence: 92.34%
```

---

## Model Training

The model training process included:

```
Dataset
   ↓
Image Preprocessing
   ↓
Data Augmentation
   ↓
Backbone (EfficientNetB0 / MobileNetV2)
   ↓
Transfer Learning
   ↓
Fine-Tuning
   ↓
Early Stopping
   ↓
Model Evaluation & Comparison
```

The dataset was divided into training and validation data for model development and evaluation.

---

## 🔮 Future Improvements

- Add more waste categories
- Increase the size and diversity of the dataset
- Improve classification of visually similar materials (e.g., Plastic, which has the lowest F1-score)
- Add real-time camera-based classification
- Add recycling and disposal recommendations
- Add confidence visualization for all seven categories

---

## 👩‍💻 Author

**Riddhima Mehra**

B.Tech – Artificial Intelligence & Machine Learning

[GitHub](https://github.com/riddhimamehra)
