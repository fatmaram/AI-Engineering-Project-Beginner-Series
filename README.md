# AI-Engineering-Project-Beginner-Series
Contains my first AI Engineering project.
#  Heart Disease Prediction using Machine Learning

##  Project Background
Heart disease remains one of the leading causes of death worldwide. Early detection can significantly improve patient outcomes and reduce mortality rates. This project applies machine learning techniques to predict the likelihood of heart disease based on patient health attributes.

---

##  Methodology
1. **Data Acquisition**  
   - Dataset sourced from Kaggle: [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

2. **Data Preprocessing**  
   - Handled missing values  
   - Applied feature scaling using `StandardScaler`  
   - Encoded categorical variables where necessary  

3. **Model Training**  
   - Logistic Regression (baseline model)  
   - K-Nearest Neighbors (KNN)  
   - Support Vector Machine (SVM)  
   - Decision Tree  
   - Random Forest  
   - XGBoost  

4. **Model Evaluation**  
   - Accuracy  
   - Precision, Recall, F1-score  
   - Confusion Matrix  
   - ROC-AUC  

5. **Comparison & Visualization**  
   - Bar charts comparing model accuracies  
   - Confusion matrix heatmaps  

---

##  Data Sources
- **Dataset:** Heart Disease Dataset by *johnsmith88* on Kaggle  
- **Features include:**  
  - Age, Sex, Chest Pain Type (cp), Resting Blood Pressure (trestbps), Cholesterol (chol), Fasting Blood Sugar (fbs), Resting ECG (restecg), Maximum Heart Rate (thalach), Exercise Induced Angina (exang), ST Depression (oldpeak), Slope, Number of Major Vessels (ca), Thalassemia (thal)  
- **Target variable:** `target` (1 = heart disease, 0 = no heart disease)

---

##  Model Design
- **Baseline:** Logistic Regression for interpretability  
- **Ensemble Models:** Random Forest and XGBoost for stronger predictive performance  
- **Train/Test Split:** 80/20 with fixed random state for reproducibility  

---

## Results and Conclusion
- This were the performance of differnt models:
    Model Name:  DTree  Results:  0.9902439024390244
    Model Name:  KNC  Results:  0.8195121951219513
    Model Name:  Random Forest  Results:  0.9878048780487806
    Model Name:  Support Machines  Results:  0.9048780487804879
    Model Name:  LR  Results:  0.8426829268292682
**Conclusion:**  
Across the models tested, Decision Tree (99.0%) and Random Forest (98.8%) achieved the highest accuracies, showing that tree‑based methods are particularly effective for this dataset. The Support Vector Machine (90.5%) also performed strongly, while Logistic Regression (84.3%) and K‑Nearest Neighbors (81.9%) lagged behind in predictive power.

Decision Tree: Highest accuracy, but may risk overfitting due to its tendency to memorize training data.

Random Forest: Nearly as accurate as the Decision Tree, but more robust and generalizable thanks to ensemble averaging.

Support Vector Machine: Balanced performance, effective at capturing complex boundaries.

Logistic Regression: Lower accuracy, but remains interpretable and useful as a baseline.

KNN: Lowest accuracy, suggesting it may not be well‑suited for this dataset without further tuning.

Overall: Tree‑based ensemble methods (Random Forest) provide the best balance of accuracy and reliability, making them the most suitable choice for heart disease prediction in this project. Logistic Regression, while less accurate, is still valuable for its interpretability in medical contexts.

---

##  How to Run
```bash
 Clone the repositories
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook
googlecolab notebook Heart_Disease_Prediction.ipynb
