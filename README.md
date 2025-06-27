# Stroke Prediction using Machine Learning

This project uses a variety of supervised machine learning models to predict the likelihood of a stroke based on patient health data. The dataset comes from a healthcare dataset containing demographic and clinical features of patients.

##  Dataset

The dataset used is:
- `healthcare-dataset-stroke-data.csv`

Key features include:
- Demographics: `gender`, `age`, `ever_married`, `Residence_type`
- Health indicators: `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`, `smoking_status`
- Target variable: `stroke` (1 if the patient had a stroke, otherwise 0)

##  Preprocessing

- Missing values in the `bmi` column are filled with the **mean**.
- Categorical variables are **label encoded**.
- Data is **split** into training and test sets using stratified sampling.
- **SMOTE** (Synthetic Minority Oversampling Technique) is applied to handle class imbalance in the training set.

##  Models Used

The following machine learning models were trained and evaluated:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Multilayer Perceptron (MLP) Neural Network**

## Evaluation Metrics

Models are evaluated on the test set using:

- **Accuracy**
- **Recall**
- **ROC AUC Score**
- **ROC Curve** visualizations

Bar plots and ROC curves are used to compare model performance.


##  Visualizations

- **ROC Curves** for all models
- **Bar Chart** comparing Accuracy, Recall, and ROC AUC across models

## Dependencies

Make sure the following libraries are installed:

```bash
pip install pandas matplotlib scikit-learn imbalanced-learn numpy
