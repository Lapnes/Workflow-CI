"""
MLProject Training Entry Point Script
======================================
Script yang dijalankan oleh MLproject MLflow.
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # MLflow autolog
    mlflow.sklearn.autolog()
    
    # Load dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "credit_scoring_preprocessing.csv")
    
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(base_dir, "..", "..", "Eksperimen_SML_Ragil-Amirzaky", "preprocessing", "credit_scoring_preprocessing.csv")
        
    df = pd.read_csv(dataset_path)
    
    X = df.drop(columns=['SeriousDlqin2yrs'])
    y = df['SeriousDlqin2yrs']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Running MLProject Training with n_estimators={args.n_estimators}, max_depth={args.max_depth}...")
    
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"MLProject Training Finished! Test Accuracy: {score:.4f}")

if __name__ == "__main__":
    main()
