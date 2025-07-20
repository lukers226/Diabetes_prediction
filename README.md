# 🩺 Diabetes Prediction using Machine Learning

This project uses a machine learning model to predict whether a person is likely to have diabetes based on medical diagnostic data. It is built using Python and the popular **K-Nearest Neighbors (KNN)** algorithm.

---

## 📌 Objective

To develop a model that can **predict diabetes** in patients using health-related attributes such as glucose level, BMI, blood pressure, etc. The model helps in early detection and supports healthcare professionals in making better decisions.

---

## ⚙️ Technologies Used

- 🐍 Python
- 📊 Pandas, NumPy
- 📈 Scikit-learn (KNN, Train/Test Split, Accuracy Score)
- 📉 Matplotlib / Seaborn (for data visualization)

---

## 🧪 Dataset

The dataset used is the **PIMA Indians Diabetes Dataset**, commonly used for diabetes classification tasks.

### Dataset Features:

| Feature             | Description                          |
|---------------------|--------------------------------------|
| Pregnancies         | Number of times pregnant             |
| Glucose             | Plasma glucose concentration         |
| BloodPressure       | Diastolic blood pressure (mm Hg)     |
| SkinThickness       | Triceps skin fold thickness (mm)     |
| Insulin             | 2-Hour serum insulin (mu U/ml)       |
| BMI                 | Body mass index (weight in kg/m²)    |
| DiabetesPedigree    | Diabetes pedigree function           |
| Age                 | Age of the person                    |
| Outcome             | 0 = Non-diabetic, 1 = Diabetic       |

---

## 🤖 Model Used: K-Nearest Neighbors (KNN)

KNN is a simple, non-parametric algorithm used for classification. It predicts the outcome based on the majority class among the K-nearest data points.

- `K` value: Tuned to achieve best accuracy
- Distance metric: Euclidean
- Accuracy achieved: _e.g., 78% (example only — depends on your training)_

---

## 📊 Results & Evaluation

- Accuracy: ✅ `XX%` (add your model accuracy here)
- Confusion Matrix and Classification Report used for evaluation
- Visuals: Feature correlation heatmaps, pairplots

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
pip install -r requirements.txt
jupyter notebook diabetes_prediction.ipynb


