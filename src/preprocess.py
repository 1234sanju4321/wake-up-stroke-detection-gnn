import pandas as pd

def preprocess(df):
    df = df.dropna()
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/raw/stroke_data.csv")
    processed = preprocess(df)
    processed.to_csv("data/processed/stroke_processed.csv", index=False)
    print("Preprocessing completed")
