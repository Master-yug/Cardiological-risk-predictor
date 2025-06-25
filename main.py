#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 20:37:27 2025

@author: master-yug
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("Cardiovascular_Disease_Dataset.csv")

# Drop patientid if it exists(backup step so that code doesnt blow)
if 'patientid' in df.columns:
    df.drop('patientid', axis=1, inplace=True)

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=9,metric='braycurtis')
knn.fit(X_train, y_train)

print("\nEnter the following values for prediction:")

user_input = []
feature_names = X.columns.tolist()

for feature in feature_names:
    val = float(input(f"{feature}: "))
    user_input.append(val)

user_scaled = scaler.transform([user_input])

prediction = knn.predict(user_scaled)[0]
probability = knn.predict_proba(user_scaled)[0][prediction]
print('KNN score is:',knn.score(X_test,y_test))

if prediction == 1:
    print(f"\n Likely to have heart disease (Confidence: {probability:.2f})")
else:
    print(f"\n Not likely to have heart disease (Confidence: {probability:.2f})")

