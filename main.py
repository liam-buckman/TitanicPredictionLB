# Import necessary libraries
import pandas as pd
import numpy as np 
import warnings
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv('Train.csv')

# ----------------------------
# Cleaning up the dataset
# ----------------------------

# Fill missing 'Age' values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing 'Embarked' values with the mode (most frequent value)
df['Embarked'].fillna(df['Embarked'].mode(0), inplace=True)

# Fill missing 'Fare' values with the median
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Convert 'Sex' from string to numerical (0 = male, 1 = female)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert 'Embarked' from string to numerical (C = 0, Q = 1, S = 2)
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# ----------------------------
# Feature selection and target
# ----------------------------

# Define the input features to use
Important_Features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']

# Define feature matrix X and target variable Y
X = df[Important_Features]
Y = df['Survived']

# ----------------------------
# Train-test split
# ----------------------------

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random.randint(1, 100))

# Save the indices of the test set (optional)
tests = X_test.index

# Save the training data to a CSV for inspection/debugging
train_data = X_train.copy()
train_data['Survived'] = y_train
train_data.to_csv('training_data.csv', index=False)

# ----------------------------
# Model 1: Logistic Regression
# ----------------------------

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions using the logistic regression model
y_pred = model.predict(X_test)

# Count how many passengers were predicted to survive and die
survived = 0
died = 0
for i in range(len(y_pred)):
    if y_pred[i]:
        survived += 1
    else: 
        died += 1

print('model 1 done')

# Prepare values for plotting
values = [survived, died]
names = ['survived', 'died']

# Plot predictions from Logistic Regression
plt.subplot(131)
plt.bar(names, values)
plt.xlabel('Outcome of model 1')
plt.ylabel('amount of passengers')

# ----------------------------
# Model 2: Random Forest Classifier
# ----------------------------

# Initialize and train the Random Forest model
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model2.fit(X_train, y_train)

# Make predictions using the Random Forest model
y_pred2 = model2.predict(X_test)

# Count how many passengers were predicted to survive and die
survived2 = 0
died2 = 0
for i in range(len(y_pred2)):
    if y_pred2[i]:
        survived2 += 1
    else: 
        died2 += 1

print('model 2 done')

# Prepare values for plotting
values2 = [survived2, died2]
names = ['survived', 'died']

# ----------------------------
# Actual outcomes from test set
# ----------------------------

# Count actual survival values in the test set
died3 = 0
survived3 = 0
for i in range(len(y_test)):
    if y_test.iloc[i]:
        survived3 += 1
    else: 
        died3 += 1

# Prepare actual values for plotting
values3 = [survived3, died3]

# ----------------------------
# Accuracy evaluation
# ----------------------------

# Count correct and incorrect predictions for model 1
correct1 = 0
wrong1 = 0
for i in range(len(y_test)):
    if y_pred[i] == y_test.iloc[i]:
        correct1 += 1
    else: 
        wrong1 += 1

# Print accuracy for model 1
print(f'Model one Accuracy: {correct1 / (wrong1 + correct1) * 100} %')

# Count correct and incorrect predictions for model 2
correct2 = 0
wrong2 = 0
for i in range(len(y_test)):
    if y_pred2[i] == y_test.iloc[i]:
        correct2 += 1
    else: 
        wrong2 += 1

# Print accuracy for model 2
print(f'Model two Accuracy: {correct2 / (wrong2 + correct2) * 100} %')

# ----------------------------
# Plotting
# ----------------------------

# Plot predictions from Random Forest
plt.subplot(132)
plt.bar(names, values2)
plt.xlabel('Outcome of model 2')
plt.ylabel('amount of passengers')

# Plot actual test outcomes
plt.subplot(133)
plt.bar(names, values3)
plt.xlabel('Actual Outcome')
plt.ylabel('amount of passengers')

# Show all the plots
plt.show()