import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.copy()

    # Drop irrelevant columns
    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Encode categorical columns
    label_encoders = {}
    for col in ['Sex', 'Embarked']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

def split_data(df, test_size, val_size, random_state):
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test

def save_data(X_train, X_val, X_test, y_train, y_val, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_val.to_csv(f'{output_dir}/X_val.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_val.to_csv(f'{output_dir}/y_val.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)

def main():
    config = load_config()
    df = load_data(config['data']['raw_path'])
    df_clean, _ = clean_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df_clean, 
        config['data']['test_size'], 
        config['data']['val_size'], 
        config['data']['random_state']
    )
    save_data(X_train, X_val, X_test, y_train, y_val, y_test, config['data']['processed_dir'])
    print("✅ Titanic data preprocessing complete and saved.")

if __name__ == "__main__":
    main()
