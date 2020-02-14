##################################################
"""This code was written for the WiMLDS monthly meetup on February 20, 2020.
This code presents examples and comparison of using SHAP and Lime for machine
learning interpretability"""
##################################################
# Python Version: 3.6.8
# Data Source: https://archive.ics.uci.edu/ml/datasets/Student+Performance
##################################################

# Library imports
import shap
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# print the JS visualization code to the notebook
shap.initjs()

# ----------------
# Load Data
# ----------------
student_data = pd.read_csv("student_data_prepped.csv")

# Look at the distribution of passing grades
plt.hist(student_data.Pass)

# ----------------
# Classification
# ----------------

# Using XGBoost
X = student_data.drop(['G3', 'Pass'], axis=1)
y = student_data['Pass']

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Initialize and fit classifier
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions on test set
y_pred = xgb_model.predict(X_test)

# Print prediction results
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Print a classification report to view accuracy metrics
print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))

# -----------
# SHAP
# -----------

# explain the model's predictions using SHAP values
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_train)

# visualize the first prediction's explanation 
# Use link='logit' to see the probability of predicting 1, rather than seeing log-odds
shap.force_plot(explainer.expected_value, shap_values[0, :], X_train.iloc[0, :], link='logit')

# visualize the second prediction's explanation 
shap.force_plot(explainer.expected_value, shap_values[2, :], X_train.iloc[2, :], link='logit')

# visualize the training set predictions
shap.force_plot(explainer.expected_value, shap_values, X_train, link='logit')

# SHAP summary plot
shap.summary_plot(shap_values, X_train)

# Traditional variable importance plot
shap.summary_plot(shap_values, X_train, plot_type="bar")

# Dependence plots showing the impact of individual and interaction features on SHAP values
shap.dependence_plot("G2", shap_values, X_train, interaction_index=None)
shap.dependence_plot("G2", shap_values, X_train, interaction_index="sex_F")

# Decision plot
select = range(20)
features = X_test.iloc[select]
features_display = X.loc[features.index]

# View decision plot for first 10 observations
shap.decision_plot(explainer.expected_value, shap_values[:10,:], features_display, link="logit")

# View decision plot for first observation
shap.decision_plot(explainer.expected_value, shap_values[0,:], features_display, link="logit")

# View decision plot for third observation
shap.decision_plot(explainer.expected_value, shap_values[2,:], features_display, link="logit")
