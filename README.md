# 🩺 Breast Cancer Detection using Artificial Neural Networks (ANN)  
![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?logo=python&logoColor=white) ![Keras](https://img.shields.io/badge/Uses-Keras%20%26%20TensorFlow-orange?logo=tensorflow&logoColor=white) ![MIT License](https://img.shields.io/badge/License-MIT-green)

<img src="https://img.icons8.com/external-flaticons-lineal-color-flat-icons/64/000000/external-breast-cancer-world-cancer-day-flaticons-lineal-color-flat-icons.png" width="64" align="right"/>

A complete end-to-end project for detecting breast cancer (benign/malignant) from clinical data using an Artificial Neural Network (ANN) built with Keras & TensorFlow.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Demo](#demo)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Key Steps in the Notebook](#key-steps-in-the-notebook)
- [Results](#results)
- [References](#references)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## 🚀 Project Overview

Breast cancer is one of the leading causes of cancer-related deaths in women worldwide. Early detection is crucial for effective treatment. This project leverages modern machine learning—specifically, an Artificial Neural Network (ANN)—to classify tumors as **benign** or **malignant** based on clinical features.

The notebook walks through every step: data cleaning, exploration, visualization, model building, evaluation, and more. If you want to learn how to use deep learning for tabular medical data, you’re in the right place!

---

## ✨ Features

- 📈 **Data Exploration:** Visual and statistical analysis of the dataset  
- 🧹 **Data Preprocessing:** Cleaning, encoding, scaling  
- 🧠 **ANN Model:** Built from scratch using Keras (TensorFlow backend)  
- 🏆 **Model Evaluation:** Accuracy, confusion matrix, classification report  
- 📊 **Visualization:** Training curves, result charts  
- 🖱️ **Easy to Use:** Just open the notebook and run cell-by-cell!

---

## 🎥 Demo

![Breast Cancer ANN Demo](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYTk3ZTc5MDExNThlMjYwNTcxZmQ3YmI4Y2U5MTQ2YjAyODFkNGRjMiZjdD1n/WoWm8YzFQJg5i/giphy.gif)

---

## 🛠️ Getting Started

**Clone the repo:**

```bash
git clone https://github.com/yourusername/Breast-Cancer-Detection-ANN.git
cd Breast-Cancer-Detection-ANN
```

## 🛠️ Install the Dependencies

```bash
pip install -r requirements.txt
```
or install directly:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## 📥 Download the Dataset

Place your `data.csv` (Breast Cancer dataset) in the **project root** directory.

You can download the dataset from:  
🔗 [Kaggle - Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)


## 📦 Usage

### 1. Open the Notebook

- **Jupyter Notebook:**  
  Open `Breast_Cancer_Detection_uisng_ANN.ipynb` in Jupyter Notebook.

- **Google Colab (Recommended):**  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

### 2. Run All Cells

- Follow the notebook **step-by-step**.
- All **code**, **plots**, and **metrics** will be displayed.
- Model **accuracy**, **loss curves**, and **evaluation reports** will appear automatically.

## 🔑 Key Steps in the Notebook

### 1. Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

### 2. Load and Inspect Dataset

```python
df = pd.read_csv('data.csv')
print(df.info())
df.head()
```

### 3. Data Cleaning & Preprocessing

```python
df = df.drop(['id'], axis=1)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
```

### 4. Visualization

```python
sns.countplot(x='diagnosis', data=df)
plt.title('Diagnosis Class Distribution')
plt.xlabel('Diagnosis (1 = Malignant, 0 = Benign)')
plt.ylabel('Count')
plt.show()
```

### 5. Train/Test Split & Scaling

```python
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 6. Build & Train the ANN

```python
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test)
)
```

### 7. Model Evaluation

```python
# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy over Epochs')
plt.legend()
plt.show()

# Predict & metrics
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## 📊 Results

- **Validation Accuracy:** ~97–99% (depends on train-test split)
- **Confusion Matrix:** Displays True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
- **Classification Report:** Includes Precision, Recall, F1-score, and Support for each class

> ⚠️ This model shows high performance but should be validated thoroughly before real-world deployment. Additional techniques like cross-validation, hyperparameter tuning, and domain expert evaluation are recommended.


## 🔗 References

- [UCI Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Keras Documentation](https://keras.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- Witten, I.H., Frank, E., Hall, M.A., & Pal, C.J. (2016). *Data Mining: Practical Machine Learning Tools and Techniques* (4th ed.). Morgan Kaufmann. [Link](https://www.elsevier.com/books/data-mining/witten/978-0-12-804291-5)

---

## 🪪 License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for more details.

---

## 🙌 Acknowledgments

- UCI & Kaggle for open-access datasets.
- Open-source contributors around the world for making such projects possible.

## 📬 Contact

For questions, feedback, or contributions, feel free to:

- 🛠️ **Raise an Issue**  
- 📤 **Open a Pull Request**

Let's collaborate and improve this project together.

> Let’s beat cancer with code! 🩷



