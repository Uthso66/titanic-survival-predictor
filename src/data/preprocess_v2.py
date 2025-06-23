import os
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from {path}")
    return config

def load_data(path):
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Data shape: {df.shape}")
    return df

def clean_data(df):
    df = df.copy()
    initial_shape = df.shape

    victim_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df.drop(columns=victim_columns, inplace=True)
    logger.info(f"Eliminated {len(victim_columns)} columns")

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    logger.info(f"Filled missing values- Age and Embarked")

    label_encoders = {}
    categorical_cols = ['Sex', 'Embarked']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        logger.info(f"Encoded {col}: {list(le.classes_)}")

    final_shape = df.shape
    logger.info(f"Data cleaned: {initial_shape} -> {final_shape}")
    return df, label_encoders

def split_data(df, test_size, val_size, random_state):
    """Split data like dealing card at a high-stakes table"""
    X = df.drop(columns=['Survived'])
    
