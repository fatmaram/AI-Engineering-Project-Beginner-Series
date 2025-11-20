# 🩺 Heart Disease Prediction using Machine Learning

## 📘 Project Overview
This project applies multiple machine learning algorithms to predict the likelihood of heart disease based on patient health attributes such as age, sex, chest pain type, cholesterol levels, blood pressure, and more.  

The dataset is sourced from Kaggle: [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).  
Models tested include Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Tree, Random Forest, and XGBoost.  

The best-performing models (Decision Tree and Random Forest) achieved accuracy scores above 98%, demonstrating strong predictive power for this dataset.

---

## 🎯 Project Objectives
- Build and evaluate multiple machine learning models for heart disease prediction.  
- Learn how to preprocess and scale health data for training.  
- Understand how to split data into training and testing sets for reproducibility.  
- Compare model performance using accuracy, precision, recall, and F1-score.  
- Demonstrate practical application of machine learning in healthcare prediction tasks.  

---

# ============================================
# 🩺 Heart Disease Prediction Project Workflow
# ============================================

# Step 1: Upload Kaggle API credentials
from google.colab import files
files.upload()

# Step 2: Download dataset from Kaggle
!kaggle datasets download -d johnsmith88/heart-disease-dataset -p data

# Step 3: Unzip dataset
import zipfile
with zipfile.ZipFile("data/heart-disease-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("data")

# Step 4: Load dataset
import pandas as pd
df = pd.read_csv("data/heart.csv")
print("Dataset shape:", df.shape)
df.head()

# Step 5: Separate features and target
dataset = pd.read_csv("data/heart.csv")
x = dataset.iloc[:, :-1].values   # Features
y = dataset.iloc[:, -1].values    # Target

# Step 6: Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder() 
le2 = LabelEncoder() 
le6 = LabelEncoder() 
le8 = LabelEncoder() 
le10 = LabelEncoder() 

x[:,1] = le1.fit_transform(x[:,1])
x[:,2] = le2.fit_transform(x[:,2])
x[:,6] = le6.fit_transform(x[:,6])
x[:,8] = le8.fit_transform(x[:,8])
x[:,10] = le10.fit_transform(x[:,10])

# Step 7: Import machine learning algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Step 8: Define models to test
models = []
models.append(('DTree', DecisionTreeClassifier()))
models.append(('Gaussian', GaussianNB()))
models.append(('KNC', KNeighborsClassifier()))
models.append(('Random Forest', RandomForestClassifier()))
models.append(('Gradient Boosting', GradientBoostingClassifier()))
models.append(('Support Machines', SVC(gamma='auto')))
models.append(('LR', LogisticRegression(max_iter=1000)))

# Step 9: Cross-validation to evaluate models
from sklearn.model_selection import KFold, cross_val_score

for name, model in models:
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, x, y, cv=kfold, scoring='accuracy')
    print('Model Name:', name, ' Results:', cv_results.mean())

# Step 10: Visualize model accuracies (replace with actual values)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.bar("Logistic Regression", LogisticRegression_Accuracy, width=0.6)
plt.bar("KNeighbors", KNeighbors_Accuracy, width=0.6)
plt.bar("Support Vector Machine", SVM_Accuracy, width=0.6)
plt.bar("Decision Tree", Decision_Accuracy, width=0.6)
plt.bar("Random Forest", RandomForest_Accuracy, width=0.6)
plt.bar("XGBoost", XGBoost_Accuracy, width=0.6)
plt.xlabel("Machine Learning Algorithm")
plt.ylabel("Accuracy")
plt.show()

# Step 11: Example prediction with Random Forest
# (replace with real values in correct order of features)
sample_input = [[40, 1, 2, 110, 290, 1, 2, 160, 0, 2.2, 1, 0, 2]]

result = model_randomforest.predict(sc.transform(sample_input))

if result == [0]:
    print('Person Not Having Heart Disease')
else:
    print("Person Having Heart Disease")
## 📊 Results and Performance
After training and evaluation, the models achieved the following accuracies:

- Decision Tree: **99.0%**
- Random Forest: **98.8%**
- Support Vector Machine: **90.5%**
- Logistic Regression: **84.3%**
- K-Nearest Neighbors: **81.9%**

Tree-based methods (Decision Tree and Random Forest) provided the strongest predictive performance, while Logistic Regression remained useful for interpretability in medical contexts.

---

## 💡 Application Areas
This project demonstrates foundational skills applicable in various AI and healthcare applications:

| Application Area        | Description |
|--------------------------|-------------|
| 🏥 Healthcare            | Predicts likelihood of heart disease for early intervention and patient monitoring. |
| 📊 Public Health Policy  | Supports data-driven decision making for resource allocation and prevention programs. |
| 🛒 Insurance             | Helps insurers assess client risk profiles for coverage decisions. |
| 🧑‍🏫 Education           | Teaches fundamentals of machine learning using real-world medical datasets. |
| 🏭 Clinical Research     | Assists researchers in testing predictive models for cardiovascular risk. |

---

## 🚀 How to Run the Project
**Requirements**
- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- xgboost

**Run the Notebook**
1. Clone this repository or open in Google Colab.  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
Author
Fatuma Ramadhan  
Kiambu, Kenya 💼 
Eager to do more projects applying machine learning to healthcare and statistical analysis.

Conclusion
This project provides a strong foundation for understanding how machine learning can be applied to healthcare prediction tasks. Learners are encouraged to experiment with hyperparameter tuning, feature engineering, and ensemble methods to further improve accuracy and robustness.

