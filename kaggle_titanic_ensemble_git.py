import pandas as pd 
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Open files and do a bit of parsing
df = pd.read_csv('titanic_train.csv')
test = pd.read_csv('kaggle_titanic_test.csv')
pass_numb = test['PassengerId']
y = df['Survived'] 

# Set my the features I want to use.
features = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
test_features = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Replace NaN values and change the categorical info into numbers.
features['Age'].fillna(features['Age'].mean(), inplace=True)
features['Fare'].fillna(features['Fare'].value_counts().idxmax(), inplace=True)
features = pd.get_dummies(features, drop_first=True)

test_features['Age'].fillna(test_features['Age'].mean(), inplace=True)
test_features['Fare'].fillna(test_features['Fare'].value_counts().idxmax(), inplace=True)
test_features = pd.get_dummies(test_features, drop_first=True)

# Create pipeline and scale the data 
# I have only tuned one of the models at this time.
pipeline_GBC = make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=83, subsample=0.926, min_samples_split=4))
pipeline_SVC = make_pipeline(StandardScaler(), SVC(probability=True))
pipeline_XGB = make_pipeline(StandardScaler(), XGBClassifier())

# Fit all the models
pipeline_GBC.fit(features, y)
pipeline_SVC.fit(features, y)
pipeline_XGB.fit(features, y)

# Return the probablities from each model.
SVC_prob = pd.DataFrame(pipeline_SVC.predict_proba(test_features))
GBC_prob = pd.DataFrame(pipeline_GBC.predict_proba(test_features))
XGB_prob = pd.DataFrame(pipeline_XGB.predict_proba(test_features))

# Create new dataframe of model predictions.
p_of_one =  pd.concat([SVC_prob.loc[:,1], GBC_prob.loc[:,1], XGB_prob.loc[:,1]], keys=['SVC_1', 'GBC_1', 'XGB_1'], axis=1)

# Find the evenly weighted probablity of survival.
unweighted_p_of_one = (p_of_one['SVC_1'] +  p_of_one['GBC_1'] + p_of_one['XGB_1']) / 3

def decider(x):
	if x > .5:
		return 1
	else:
		return 0

# Convert probablity to binary.
one_or_zero = unweighted_p_of_one.apply(decider)

y_test.reset_index(drop=True, inplace=True)

# Create the final dataframe for submission to Kaggle.com
final = pd.concat([pass_numb, one_or_zero], axis=1, keys=['PassengerId', 'Survived'])

final.to_csv('kaggle_titanic_SVC_GBC_XGB_ensemble_tuned.csv', index = False)
