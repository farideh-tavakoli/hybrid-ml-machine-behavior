# A Hybrid Clustering–Classification Approach for Analyzing Machine Usage Patterns in Industrial IoT Data

A hybrid machine learning framework for analyzing industrial IoT data to identify machine behavior patterns, detect underperforming machines, and support predictive maintenance.  
The approach combines **K-Means clustering** for behavioral segmentation and **SVM-based classification** for recognition of these patterns in new data.

**Run the notebooks on Google Colab:**
- [Clustering Notebook](https://colab.research.google.com/github/yourusername/machine-behavior-hybrid-ml/blob/main/clustering.ipynb)
- [Classification Notebook (sub_3)](https://colab.research.google.com/github/yourusername/machine-behavior-hybrid-ml/blob/main/classification_sub_3.ipynb)



## Clustering Phase
Implemented in **`clustering.ipynb`**.  
This step groups machines with similar operational behaviors using K-Means and visualizes clusters using PCA.  
The resulting cluster labels form the foundation for supervised learning.




## Classification Phase
Implemented in **`classification_sub_3.ipynb`**, focusing on the subtechnology **`sub_3`**.  
Trains and evaluates multiple classifiers (SVM, Decision Tree, KNN) using stratified 5-fold cross-validation, optimizing the **macro F1-score**.

The final saved model is:

`model/svm_best_pipeline.pkl`

This model includes both `MinMaxScaler` and the trained SVM classifier.



## Using the Saved Model for Predictions on New Data

### 1. Load the saved model
```python
import joblib

model_path = "model/svm_best_pipeline.pkl"
svm_loaded = joblib.load(model_path)
```


### 2. Load your new data

```python
import pandas as pd

df = pd.read_csv("path_to_your_new_data.csv")
```

### 3. Preprocessing steps

Perform the same cleaning and filtering steps used during training:
```python
# Columns to keep
columns_to_keep = [
    'EXE_TotalDuration_min', 'FAIL_TotalDuration_min',
    'READY_TotalDuration_min', 'POWER_OFF_TotalDuration_min',
    'N_Change_EXE_READY', 'N_Change_EXE_FAIL', 'N_Change_EXE_POWER_OFF',
    'N_Change_READY_EXE', 'N_Change_READY_FAIL', 'N_Change_READY_POWER_OFF',
    'N_Change_FAIL_READY', 'N_Change_FAIL_EXE', 'N_Change_FAIL_POWER_OFF',
    'N_Change_POWER_OFF_EXE', 'N_Change_POWER_OFF_READY',
    'N_Change_POWER_OFF_FAIL', 'Serial Number', 'Date',
    'Subtechnology_Name_ITA', 'Shipment Date', 'Manufacturing Date',
    'Cluster_Label'
]


# Impute NULLs in numeric columns with 0
numeric_cols = [
    'EXE_TotalDuration_min', 'FAIL_TotalDuration_min', 'READY_TotalDuration_min', 'POWER_OFF_TotalDuration_min',
    'N_Change_EXE_READY', 'N_Change_EXE_FAIL', 'N_Change_EXE_POWER_OFF',
    'N_Change_READY_EXE', 'N_Change_READY_FAIL', 'N_Change_READY_POWER_OFF',
    'N_Change_FAIL_READY', 'N_Change_FAIL_EXE', 'N_Change_FAIL_POWER_OFF',
    'N_Change_POWER_OFF_EXE', 'N_Change_POWER_OFF_READY', 'N_Change_POWER_OFF_FAIL']

df[numeric_cols] = df[numeric_cols].fillna(0)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
columns_to_check = [
    'EXE_TotalDuration_min', 'FAIL_TotalDuration_min',
    'READY_TotalDuration_min', 'POWER_OFF_TotalDuration_min']
df = df[~(df[columns_to_check] == 0).all(axis=1)]
columns_null_to_drop = ["Shipment Date", "Subtechnology_Name_ITA"]
df = df.dropna(subset=columns_null_to_drop, how='any')
df = df.drop_duplicates()
specific_subtechnologies = ['sub_3']
df = df[df['Subtechnology_Name_ITA'].isin(specific_subtechnologies)]
```

### 4. Prepare feature and target columns
```python
X = df.drop(columns=[
    'Serial Number', 'Date', 'Cluster_Label', 'Shipment Date', 'Manufacturing Date',
    'Subtechnology_Name_ITA', 'N_Change_READY_EXE', 'N_Change_FAIL_READY',
    'N_Change_FAIL_EXE', 'N_Change_POWER_OFF_READY', 'N_Change_POWER_OFF_FAIL'
])
y = df["Cluster_Label"]

print(f'Feature Columns: {X.columns.tolist()}')
print(f'Target Column: {y.name}')
```

### 5. (Optional) Split data for hold-out testing
```python
from sklearn.model_selection import train_test_split, StratifiedKFold

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### 6. Predict the labels
```python
# The pipeline handles scaling automatically
y_pred = svm_loaded.predict(X)

# Attach predictions to the DataFrame
df["Predicted_Cluster"] = y_pred

# Save or view results
df.to_csv("predicted_new_data.csv", index=False)
print("Predictions saved to predicted_new_data.csv")
```


## Notes
- The project follows the CRISP-DM methodology.
- Evaluation metric: F1-score (macro/weighted) to balance precision and recall.
- Designed for industrial IoT datasets containing machine state durations and transition counts.
- Developed and tested in Python 3.10 with scikit-learn, pandas, and matplotlib.


## Repository Structure

```bash
machine-behavior-hybrid-ml/
├── clustering.ipynb              
├── classification_sub_3.ipynb 
│
├── model/
│   └── svm_best_pipeline.pkl
│
├── dataset/
│   └── cleaned_dataset.csv   
│
├── clustering_output_dbscan
│                
├── clustering_output_kmeans             
│
├── requirements.txt
├── README.md         
└── LICENSE             
```

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/machine-behavior-hybrid-ml.git
cd machine-behavior-hybrid-ml
pip install -r requirements.txt
```

## Author

Developed by **[Farideh Tavakoli](https://github.com/farideh-tavakoli)** 

[farideh.tavakoli@studio.unibo.it](mailto:farideh.tavakoli@studio.unibo.it)  


If you use this project or build upon it, please consider citing or referencing the repository.  
Contributions, suggestions, and issues are always welcome!