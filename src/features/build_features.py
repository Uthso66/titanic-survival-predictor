import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import yaml

def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_processed_data(processed_dir):
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    X_val = pd.read_csv(os.path.join(processed_dir, 'X_val.csv'))
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    return X_train, X_val, X_test

def build_feature_pipeline(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    return preprocessor

def transform_and_save_features(preprocessor, X_train, X_val, X_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)

    joblib.dump(preprocessor, os.path.join(output_dir, 'feature_pipeline.pkl'))
    
    np.save(os.path.join(output_dir, 'X_train_transformed.npy'), X_train_transformed)
    np.save(os.path.join(output_dir, 'X_val_transformed.npy'), X_val_transformed)
    np.save(os.path.join(output_dir, 'X_test_transformed.npy'), X_test_transformed)
    
    print("✅ Feature engineering complete and saved.")

def main() :
    config = load_config()
    processed_dir = config['data']['processed_dir']
    features_dir = config['data']['features_dir']
    numeric_features = config['features']['numeric']
    categorical_features = config['features']['categorical']

    X_train, X_val, X_test = load_processed_data(processed_dir)
    preprocessor = build_feature_pipeline(numeric_features, categorical_features)
    transform_and_save_features(preprocessor, X_train, X_val, X_test, features_dir)

if __name__ == '__main__':
    main()
