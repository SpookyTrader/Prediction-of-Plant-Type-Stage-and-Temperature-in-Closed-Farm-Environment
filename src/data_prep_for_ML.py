import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import yaml

def train_test_data(df, predict_target):
    # function for loading and read in config file
    def load_config(config_file):
        with open(config_file) as file:
            config = yaml.safe_load(file)
        return config
    # Config file with specifications of test size and random state
    config = load_config("src/config.yaml")
    
    if predict_target == 'Plant Type-Stage':
        # Convert string values of Y variable into integer.
        col_name = df['Plant Type-Stage'].unique().tolist()
        col_name.sort()
        df['Plant Type-Stage'] = df['Plant Type-Stage'].apply(lambda x: col_name.index(x))

    # Convert string values of categorical X variables into integer by one-hot encoding.
    onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
    if predict_target == 'Plant Type-Stage':
        X_encoded = onehot_encoder.fit_transform(df[['System Location Code', 'Previous Cycle Plant Type']])
    elif predict_target == 'Temperature':
        X_encoded = onehot_encoder.fit_transform(df[['System Location Code', 'Previous Cycle Plant Type', 'Plant Type-Stage']])
    X_encoded_df = pd.DataFrame(X_encoded)
    X_encoded_df.columns = onehot_encoder.get_feature_names_out()
    df = pd.concat([df, X_encoded_df], axis=1)

    #Separate Y varaibles from X variables, and remove categorical columns that are not encoded.
    if predict_target == 'Plant Type-Stage':
        X = df.drop(['System Location Code', 'Previous Cycle Plant Type', 'Plant Type','Plant Stage','Plant Type-Stage'], axis=1)
        y = df['Plant Type-Stage']
    elif predict_target == 'Temperature':
        X = df.drop(['System Location Code', 'Previous Cycle Plant Type', 'Plant Type','Plant Stage','Plant Type-Stage','Temperature Sensor (°C)'], axis=1)
        y = df['Temperature Sensor (°C)']

    # Split X an Y into training and test sets.
    if predict_target == 'Plant Type-Stage':
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=config['test_size'], random_state=config['random_state'])
    elif predict_target == 'Temperature':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], random_state=config['random_state'])

    # Scale all numerical variables into the same scale.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    return X_train, X_test, y_train, y_test
    