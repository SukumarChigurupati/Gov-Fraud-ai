import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw dataset."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess raw data."""

    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing numeric values with mean
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Fill missing categorical values with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale numeric features."""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


def save_processed(df: pd.DataFrame, path: str):
    """Save processed dataset."""
    df.to_csv(path, index=False)


def run_etl():
    raw_path = "data/medicare_fraud.csv"
    processed_path = "data/processed_fraud.csv"

    print("Loading raw data...")
    df = load_raw_data(raw_path)

    print("Cleaning data...")
    df = clean_data(df)

    print("Scaling features...")
    df = scale_features(df)

    print("Saving processed data...")
    save_processed(df, processed_path)

    print("✅ ETL complete! File saved at:", processed_path)


# ✅ SAFE EXIT — ETL only runs when executed directly, NOT when imported by Streamlit
if __name__ == "__main__":
    print("Running ETL locally...")
    run_etl()
