import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.multioutput import ClassifierChain
import joblib

# Load data
data = pd.read_csv("transport_dataset.csv")
print(data.head())

# Perform EDA (e.g., data.describe(), data.corr(), data.isnull().sum(), etc.)

# Data preprocessing
# Encode categorical features
encoder = LabelEncoder()
data['Size'] = encoder.fit_transform(data['Size'])

# Split the data into train and test sets
X = data.drop(['Product_Id', 'Air', 'Road', 'Rail', 'Sea'], axis=1)
y = data[['Air', 'Road', 'Rail', 'Sea']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data (if necessary)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Approach 0 - Naive Independent Models
# Implement LightGBM classifiers for each target label and evaluate their performance

target_labels = ['Air', 'Road', 'Rail', 'Sea']
f1_scores = []

for label in target_labels:
    lgbm = LGBMClassifier()
    lgbm.fit(X_train, y_train[label])
    y_pred = lgbm.predict(X_test)
    f1 = f1_score(y_test[label], y_pred)
    f1_scores.append(f1)
    print(f"{label} - F1 Score: {f1}")

mean_f1_score = np.mean(f1_scores)
print(f"Mean F1 Score: {mean_f1_score}")

# Approach 1 - Classifier Chains
# Implement ClassifierChain with LightGBM and evaluate its performance
chain = ClassifierChain(LGBMClassifier())
chain.fit(X_train, y_train)
y_pred_chain = chain.predict(X_test)
chain_f1_score = f1_score(y_test, y_pred_chain, average='weighted')
print(f"Classifier Chain (LightGBM) - Weighted F1 Score: {chain_f1_score}")

# Approach 2 - Natively Multilabel Models
# Implement Extra Trees and Neural Networks classifiers and evaluate their performance

# Extra Trees Classifier
et = ExtraTreesClassifier(n_estimators=100, random_state=42)
et.fit(X_train, y_train)
y_pred_et = et.predict(X_test)
et_f1_score = f1_score(y_test, y_pred_et, average='weighted')
print(f"Extra Trees - Weighted F1 Score: {et_f1_score}")

# Neural Network Classifier (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
mlp_f1_score = f1_score(y_test, y_pred_mlp, average='weighted')
print(f"Neural Network (MLP) - Weighted F1 Score: {mlp_f1_score}")

# Approach 3 - Multilabel to Multiclass Approach
# Combine different combinations of labels into a single target label
y_combined = y.apply(lambda row: ''.join(row.astype(str)), axis=1)
encoder_combined = LabelEncoder()
y_combined = encoder_combined.fit_transform(y_combined)
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X, y_combined, test_size=0.2, random_state=42)

# Train a LightGBM classifier on the combined labels
lgbm_combined = LGBMClassifier()
lgbm_combined.fit(X_train_combined, y_train_combined)
y_pred_combined = lgbm_combined.predict(X_test_combined)

# Evaluate model performance using f1 score, precision, and recall
f1_combined = f1_score(y_test_combined, y_pred_combined, average='weighted')
precision_combined = precision_score(y_test_combined, y_pred_combined, average='weighted')
recall_combined = recall_score(y_test_combined, y_pred_combined, average='weighted')

print(f"Multilabel to Multiclass (LightGBM) - Weighted F1 Score: {f1_combined}")
print(f"Multilabel to Multiclass (LightGBM) - Weighted Precision Score: {precision_combined}")
print(f"Multilabel to Multiclass (LightGBM) - Weighted Recall Score: {recall_combined}")

# Compare the performance of different approaches and choose the best one
approach_scores = [
    ("Naive Independent Models", mean_f1_score),
    ("Classifier Chain (LightGBM)", chain_f1_score),
    ("Extra Trees", et_f1_score),
    ("Neural Network (MLP)", mlp_f1_score),
    ("Multilabel to Multiclass (LightGBM)", f1_combined),
]

best_approach = max(approach_scores, key=lambda x: x[1])
print(f"Best Approach: {best_approach[0]} - Weighted F1 Score: {best_approach[1]}")

# Fine-tune the best approach (hyperparameter tuning and/or feature engineering)
if best_approach[0] == "Classifier Chain (LightGBM)":
    param_grid = {
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__n_estimators': [50, 100],
        'classifier__num_leaves': [31, 50],
    }
    grid_search = GridSearchCV(estimator=ClassifierChain(LGBMClassifier(), order='random', random_state=42),
                               param_grid=param_grid,
                               scoring='f1_weighted',
                               cv=5,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"Best Parameters for Classifier Chain (LightGBM): {best_params}")

    # Train the fine-tuned model
    chain_tuned = ClassifierChain(LGBMClassifier(**best_params), order='random', random_state=42)
    chain_tuned.fit(X_train, y_train)
    y_pred_chain_tuned = chain_tuned.predict(X_test)

    # Evaluate the fine-tuned model
    chain_tuned_f1_score = f1_score(y_test, y_pred_chain_tuned, average='weighted')
    print(f"Fine-tuned Classifier Chain (LightGBM) - Weighted F1 Score: {chain_tuned_f1_score}")

# Save and deploy the final best model (using joblib)
from joblib import dump, load

if best_approach[0] == "Naive Independent Models":
    # Save each individual LightGBM classifier
    for label in target_labels:
        lgbm = LGBMClassifier()
        lgbm.fit(X_train, y_train[label])
        dump(lgbm, f"{label}_lgbm_classifier.joblib")