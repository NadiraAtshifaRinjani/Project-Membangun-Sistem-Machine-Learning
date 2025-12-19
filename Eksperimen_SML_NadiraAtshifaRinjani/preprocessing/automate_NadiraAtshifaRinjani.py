import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def feature_engineering(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    return df

def drop_columns(df):
    return df.drop(columns=['street','city','statezip','country','date'])

def cap_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    return df

def handle_outliers(df):
    for col in ['price','sqft_living','sqft_lot','sqft_above','sqft_basement']:
        df = cap_outliers_iqr(df, col)
    df['bathrooms'] = df['bathrooms'].clip(upper=6)
    return df

def transform_features(df):
    df['price_log'] = np.log1p(df['price'])
    df['sqft_living_log'] = np.log1p(df['sqft_living'])
    df['sqft_lot_log'] = np.log1p(df['sqft_lot'])
    return df

def encode_features(df):
    df = pd.get_dummies(df, columns=['waterfront','view','condition','floors'], drop_first=True)
    return df

def scale_features(df):
    scaler = StandardScaler()
    cols = ['sqft_living','sqft_lot','sqft_above','sqft_basement','bedrooms','bathrooms']
    df[cols] = scaler.fit_transform(df[cols])
    return df

def preprocess_pipeline(input_path, output_path):
    df = load_data(input_path)
    df = feature_engineering(df)
    df = drop_columns(df)
    df = handle_outliers(df)
    df = transform_features(df)
    df = encode_features(df)
    df = scale_features(df)
    df.to_csv(output_path, index=False)
    print("Preprocessing selesai ✅")
    return df

if __name__ == "__main__":
    preprocess_pipeline(
        "Eksperimen_SML_NadiraAtshifaRinjani/datasetrumah_raw/datarumah.csv",
        "Eksperimen_SML_NadiraAtshifaRinjani/preprocessing/datasetrumah_preprocessing/data_preprocessed.csv"
    )

