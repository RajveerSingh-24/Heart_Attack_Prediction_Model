
# ❤️ Heart Attack Prediction Using Machine Learning

This project is a full-fledged machine learning pipeline to predict the likelihood of a heart attack based on patient data. It follows an industry-level, modular design using **Jupyter Notebook**, **MySQL**, and **multiple ML models with GridSearchCV**. The best-performing model is selected, saved, and retrained for deployment or future use.

---

## 🧠 Project Overview

- **Goal:** Predict the risk of a heart attack based on clinical parameters.
- **Input:** Tabular patient health records (CSV).
- **Output:** Classification (Heart Attack Risk: Yes/No).

---

## 🔧 Technologies Used

- Python (Jupyter Notebook)
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn (GridSearchCV, classifiers)
- Joblib (model saving/loading)
- MySQL (data storage and retrieval)

---

## 🗃️ Dataset Description

The dataset (`heart.csv`) contains health-related attributes for patients.  
Each row represents a patient, and each column is a feature:

| Feature Name | Description |
|--------------|-------------|
| `age`        | Age of the patient |
| `sex`        | Gender (1 = male, 0 = female) |
| `cp`         | Chest pain type (categorical) |
| `trestbps`   | Resting blood pressure |
| `chol`       | Serum cholesterol (mg/dl) |
| `fbs`        | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| `restecg`    | Resting ECG results |
| `thalach`    | Maximum heart rate achieved |
| `exang`      | Exercise-induced angina |
| `oldpeak`    | ST depression |
| `slope`      | Slope of the peak exercise ST segment |
| `ca`         | Number of major vessels colored by fluoroscopy |
| `thal`       | Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect) |
| `target`     | 1 = heart attack risk, 0 = no risk |

---

## 📁 Project Structure

```
heart-attack-prediction/
│
├── data/
│   └── heart.csv                 # Raw dataset
│
├── notebooks/
│   ├── 1_data_preprocessing.ipynb          # SQL import, cleaning, split
│   ├── 2_exploratory_data_analysis.ipynb   # Visualize, correlation
│   ├── 3_model_training_gridsearch.ipynb   # GridSearchCV on multiple models
│   └── 4_best_model_retrain_eval.ipynb     # Final training + evaluation
│
├── best model/
│   └── best_model.pkl            # Saved best model (joblib)
│ 
├── models/
|  └──X_train.csv
|  └── X_test.csv
|  └── y_train.csv
|  └── y_test.csv
│
└── README.md                     # Project documentation
```

---

## 📊 Step-by-Step Pipeline

### ✅ 1. Data Preprocessing (`1_data_preprocessing.ipynb`)
- Imported the CSV into **MySQL**
- Pulled it into pandas using a **cursor**
- Checked for nulls and data types
- No missing values → no further cleaning required
- Performed **train-test split** and saved the files locally

### 📈 2. Exploratory Data Analysis (`2_exploratory_data_analysis.ipynb`)
- Visualized:
  - Age distribution
  - Chest pain types by risk
  - Cholesterol and blood pressure
  - Correlation matrix
- Key insights:
  - Strong correlation between age, `thalach`, `cp`, and target
  - Some features like `chol` had weak correlation

### 🤖 3. Model Training + Grid Search (`3_model_training_gridsearch.ipynb`)
- Used `GridSearchCV` to train:
  - Logistic Regression
  - Random Forest
  - SVM
  - KNN
- Evaluated on `X_test`
- Stored accuracy and best parameters
- Saved **best model** to `models/best_model.pkl`

### 🧠 4. Retraining and Final Evaluation (`4_best_model_retrain_eval.ipynb`)
- Loaded the saved model
- Retrained it on the full dataset (`X + y`)
- Evaluated with accuracy score, classification report, and confusion matrix
- Overwrote the saved model with the final retrained version

---

## 🧪 Model Performance (Example)

| Model              | Accuracy | Best Parameters |
|-------------------|----------|-----------------|
| Logistic Regression | 0.85     | `{'C': 1, 'solver': 'liblinear'}` |
| Random Forest       | 0.88     | `{'n_estimators': 100, 'max_depth': 8}` |
| SVM                 | 0.87     | `{'C': 1, 'kernel': 'rbf'}` |
| **Best Model**      | **0.88** | Random Forest |

---

## 🛠️ Future Work (Optional Ideas)

- ✅ Add **Streamlit** or **Flask** for deployment as a web app
- ✅ Store model predictions and user input back into MySQL
- 📈 Use **cross-validation** scores instead of just train/test split
- 🔍 Add **feature importance** or SHAP explanations
- 🤖 Try more advanced models like XGBoost or LightGBM

---

## 🧑‍💻 Author

**Rajveer Singh**  
Aspiring AI/ML Engineer | B.Tech CSE  
GitHub: [@rajveersingh](https://github.com/rajveersingh)
