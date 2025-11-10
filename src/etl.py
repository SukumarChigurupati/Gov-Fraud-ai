import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw dataset. If the file does NOT exist (Streamlit Cloud), skip ETL."""
    if not os.path.exists(path):
        print(f"⚠️ WARNING: Raw data file not found at {path}. Skipping ETL.")
        return pd.DataFrame()  # return empty df so app doesn't crash

    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess raw data."""
    if df.empty:
        return df

    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale numeric features."""
    if df.empty:
        return df

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


def save_processed(df: pd.DataFrame, path: str):
    """Save processed dataset."""
    if df.empty:
        print("⚠️ No data to save (empty). Skipping save.")
        return

    df.to_csv(path, index=False)


def run_etl():
    raw_path = "data/medicare_fraud.csv"   # ❌ Not available on Streamlit Cloud
    processed_path = "data/processed_fraud.csv"

    df = load_raw_data(raw_path)

    if df.empty:
        print("⚠️ ETL skipped because raw data does not exist.")
        return

    df = clean_data(df)
    df = scale_features(df)
    save_processed(df, processed_path)

    print("✅ ETL complete! File saved at:", processed_path)


if __name__ == "__main__":
    run_etl()
