import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import List


def clean_malformed_csv(csv_path: str) -> pd.DataFrame:
    """
    Clean up a malformed CSV file that has duplicate columns and incorrect structure.
    """
    print(f"Cleaning malformed CSV: {csv_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    
    # Check if the first row contains ticker names instead of data
    if df.iloc[0, 0] == 'RELIANCE.BO' or pd.isna(df.iloc[0, 0]):
        print("Removing first row with ticker names...")
        # Remove the first row which contains ticker names
        df = df.iloc[1:].reset_index(drop=True)
    
    # Find the base columns (without .1, .2, etc. suffixes)
    base_columns = []
    for col in df.columns:
        if '.' not in col:
            base_columns.append(col)
    
    print(f"Base columns found: {base_columns}")
    
    # If we have base columns, use only those
    if base_columns:
        df = df[base_columns]
        print(f"Using only base columns. New shape: {df.shape}")
    
    # Ensure Date column is properly formatted
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        print(f"After date cleaning: {df.shape}")
    
    # Convert numeric columns
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows where all numeric columns are NaN
    if numeric_columns:
        df = df.dropna(subset=numeric_columns[:1])  # Drop if Close is NaN
    
    print(f"Final cleaned shape: {df.shape}")
    print(f"Final columns: {df.columns.tolist()}")
    
    return df


def get_bse500_data(
    start: str,
    end: str,
    tickers: List[str],
    save_path: str = "bse500_data.csv",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for provided BSE500 tickers from Yahoo Finance.
    Returns a concatenated DataFrame with columns [Date, Open, High, Low, Close, Adj Close, Volume, Ticker].
    """
    all_data = []
    for ticker in tickers:
        try:
            print(f"Fetching {ticker}...")
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if df is None or df.empty:
                print(f"No data found for {ticker}")
                continue
            
            print(f"Raw data shape: {df.shape}")
            print(f"Raw columns: {df.columns.tolist()}")
            
            # Handle MultiIndex columns from yfinance (using working logic)
            print(f"Checking for MultiIndex: {isinstance(df.columns, pd.MultiIndex)}")
            if isinstance(df.columns, pd.MultiIndex):
                print("Detected MultiIndex columns, flattening...")
                # Flatten the MultiIndex columns - take only the first level (price type)
                df.columns = df.columns.get_level_values(0)
                print(f"Flattened columns: {df.columns.tolist()}")
            else:
                print("No MultiIndex detected - columns are already flat")
            
            df = df.dropna()
            df["Ticker"] = ticker
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Check if the index was named, if not, rename it to Date
            if df.columns[0] != 'Date':
                df.rename(columns={df.columns[0]: "Date"}, inplace=True)
            
            print(f"After processing - columns: {df.columns.tolist()}")
            print(f"Sample data for {ticker}:")
            print(df.head(2))
            
            # Ensure we have the expected columns
            expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker']
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Missing columns {missing_columns} for {ticker}")
            
            all_data.append(df)
            print(f"Successfully fetched {ticker} with {len(df)} rows")
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")
    
    if not all_data:
        print("No data fetched.")
        return pd.DataFrame()
    
    # Ensure all DataFrames have the same columns before concatenation
    if len(all_data) > 1:
        # Get the common columns from all DataFrames
        common_columns = set(all_data[0].columns)
        for df in all_data[1:]:
            common_columns = common_columns.intersection(set(df.columns))
        
        # Keep only common columns
        common_columns = list(common_columns)
        all_data = [df[common_columns] for df in all_data]
        print(f"Using common columns: {common_columns}")
    
    # Concatenate vertically (stack rows)
    combined_df = pd.concat(all_data, ignore_index=True, sort=False)
    
    # Sort by Date and Ticker for better organization
    if 'Date' in combined_df.columns and 'Ticker' in combined_df.columns:
        combined_df = combined_df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    # Final validation - ensure all columns are strings (not tuples)
    print(f"Before final cleanup - columns: {combined_df.columns.tolist()}")
    
    # Check for any tuple columns and convert them
    tuple_columns = [col for col in combined_df.columns if isinstance(col, tuple)]
    if tuple_columns:
        print(f"Warning: Found tuple columns: {tuple_columns}")
        # Force convert all columns to strings
        combined_df.columns = [str(col) for col in combined_df.columns]
        print(f"After tuple cleanup - columns: {combined_df.columns.tolist()}")
    else:
        print("✅ No tuple columns found")
    
    combined_df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Unique tickers: {combined_df['Ticker'].nunique()}")
    print(f"Final columns: {combined_df.columns.tolist()}")
    return combined_df


if __name__ == "__main__":
    # Minimal subset for quick run; replace with full BSE500 universe list
    subset = ["RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "ICICIBANK.BO"]
    start_date = "2020-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    get_bse500_data(start=start_date, end=end_date, tickers=subset, interval="1d")
