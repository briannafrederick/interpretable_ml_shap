##################################################
"""This code was written for the WiMLDS monthly meetup on February 20, 2020.
This code presents examples and comparison of using SHAP and Lime for machine
learning interpretability"""
##################################################
# Python Version: 3.6.8
# Data Source: https://archive.ics.uci.edu/ml/datasets/Student+Performance
##################################################

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# ----------------
# Load Data
# ----------------
student_data = pd.read_csv("student-mat.csv")

# Data Preprocessing
student_data.dtypes

# Add a binary column indicating whether or not the student passed their final year class
# Assume that 70% is a passing grade (14 or above)
student_data['Pass'] = 0
student_data.loc[student_data['G3'] >= 12,'Pass'] = 1

# Drop any columns with constant values
student_data = student_data.loc[:, (student_data != student_data.iloc[0]).any()] 

# Encode categorical and nominal features
"""
Numeric: age, medu, fedu, travel time, study time, failures, famrel, free time, go out, Dalc, Walc, 
              health, absences
Binary features: school, sex, address, famsize, pstatus, schoolsup, famsup, paid, extra curricular, 
                nursery, higher, internet, romantic
Nominal features: mjob, fjob, reason, guardian
"""

student_data_num = student_data[['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
                                'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']]

# First transform objects to categorical features
student_data_cat = student_data[['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup',
                                'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 
                                'romantic', 'Mjob', 'Fjob', 'reason', 'guardian']].astype('category')


# One-hot encoding for categorical variables
student_data_cat_enc = pd.get_dummies(student_data_cat)

# encode output class values as integers
y = student_data['Pass'].astype('category')

# Rebuild dataset by combining numeric, encoded categorical features, and output values
student_data_prepped = pd.concat([student_data_num, student_data_cat_enc, y], axis=1, ignore_index=False)

# Feature selection using recursive feature selection
X = student_data_prepped.drop(['Pass'], axis=1)
y = student_data_prepped['Pass']

# feature selection
model = LogisticRegression(solver='lbfgs', max_iter=500)
rfe = RFE(model, n_features_to_select=20, step=1, verbose=1)
fit = rfe.fit(X, y.values.ravel())
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

selected_columns = fit.support_
student_data_prepped = X.iloc[:,selected_columns]
# Add y column to student_data_prepped
student_data_prepped['Pass'] = y

student_data_prepped.to_csv('student_data_prepped.csv', index=False)
