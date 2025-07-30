# Capstone-Project

# ❤️ Heart Disease Prediction using Machine Learning

This project is focused on predictive analysis of heart disease using various supervised machine learning algorithms. It uses the `heart.csv` dataset to train, evaluate, and compare models.

---

## 📌 Objective

To predict the likelihood of heart disease in individuals based on clinical and demographic features using models like:
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

---

## 🗃 Dataset

File: `heart.csv`  
Each row represents a patient record with features like:
- Age, gender
- Chest pain type (`cp`)
- Cholesterol, resting blood pressure, etc.
- Target: 1 = Heart disease present, 0 = Not present

---

## 🧪 Models Evaluated

Each model is evaluated based on:
- Accuracy Score
- Average Precision Score (PR AUC)
- AUC Score (ROC AUC)
- ROC and Precision-Recall Curves
- Line chart comparing all models

---

## 🧰 Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn` (`LogisticRegression`, `SVC`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`)
- `precision_recall_curve`, `roc_curve`, `classification_report`, etc.

---

## 📊 Output Visuals

- Correlation Heatmap
- Confusion Matrices
- ROC Curve for each model
- Precision-Recall Curve for each model
- Line chart comparing Accuracy, AUC, and Average Precision

---

## ▶️ How to Run

1. Open `Heart_Disease_Model_Analysis.ipynb` in Jupyter Notebook or VS Code.
2. Ensure `heart.csv` is in the same directory.
3. Run the notebook cell by cell to:
   - Load and clean data
   - Perform EDA
   - Train and evaluate multiple models
   - Visualize performance metrics

---

## ✅ Requirements

Install required packages (if not already):

```bash
pip install pandas matplotlib seaborn scikit-learn
```

---

## 👩‍💻 Authors

- Keerthi K  
- Dharani K  
- Boopathi Raja  
- Vignesh S  
- Gopala Krishna Giridhar Parimi

---

## 📚 References

- World Health Organization (WHO) on cardiovascular diseases
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) *(if applicable)*

---

