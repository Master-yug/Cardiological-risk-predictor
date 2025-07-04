#  Early Heart Disease Detection using KNN

This project implements a machine learning pipeline to predict the **presence or absence of heart disease** using a real-world clinical dataset. We use the **K-Nearest Neighbors (KNN)** algorithm to build a predictive model that allow users to input patient data for real-time predictions.

---

##  Dataset Description

The dataset is acquired from a **multispecialty hospital in India** and Over 14 common features which makes it one of the heart disease dataset available so far for research purposes. This dataset consists of 1000 subjects with 12 clinical features. These features include:

- Age, Gender
- Chest Pain Type
- Resting Blood Pressure
- Serum Cholesterol
- Fasting Blood Sugar
- Resting Electrocardiographic Results
- Maximum Heart Rate Achieved
- Exercise-Induced Angina
- ST Depression (Oldpeak)
- Slope of the ST Segment
- Number of Major Vessels
- `target`: 0 = No heart disease, 1 = Presence of heart disease

> **License:** The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See Acknowledgements.

---

## What Model Is used???

We used **K-Nearest Neighbors (KNN)** classifier with:
- Data standardization using `StandardScaler`
- Hyperparameter tuning for `k` (number of neighbors)

---

## How Does This Work???

### Training Phase
1. Load and preprocess the dataset
2. Drop non-informative columns (like `patientid`)
3. Scale features to improve KNN distance metric performance
4. Split into training/test sets
5. Train KNN classifier

### Prediction Phase
- Accepts real-time user input from CLI
- Scales input using trained `StandardScaler`
- Predicts target label (0 or 1)
- Outputs confidence of prediction

---

## Installation

Clone the repo and install required packages:

```bash
git clone https://github.com/Master-yug/Cardiological-risk-predictor.git
cd heart-disease-knn
pip install -r requirements.txt
```

---

### License

-Code in this repository is licensed under the GNU Affero General Public License v3.0.
-Dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). You are free to share and adapt with attribution.

### Acknowledgements

-Dataset - Doppala, Bhanu Prakash; Bhattacharyya, Debnath (2021), “Cardiovascular_Disease_Dataset”, Mendeley Data, V1, doi: 10.17632/dzz48mvjht.1

-We modified the dataset for preprocessing and standardization to fit the needs of this machine learning project.
-Thanks to the Scikit-learn team
