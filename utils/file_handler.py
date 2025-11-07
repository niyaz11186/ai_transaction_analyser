"""File handling utilities for Excel and CSV operations."""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_excel(file_path: str) -> pd.DataFrame:
    """
    Load Excel file into a pandas DataFrame.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        DataFrame with transaction data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_excel(file_path)
    
    # Validate required columns
    required_columns = [
        'S No.',
        'Transaction Date',
        'Transaction Remarks',
        'Withdrawal Amount(INR)',
        'Deposit Amount(INR)',
        'Balance(INR)'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return df


def save_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        output_path: Path where CSV should be saved
    """
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics from processed DataFrame.
    
    Args:
        df: Processed DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    stats = {
        'total_transactions': len(df),
        'total_withdrawals': df['Withdrawal Amount(INR)'].sum() if 'Withdrawal Amount(INR)' in df.columns else 0,
        'total_deposits': df['Deposit Amount(INR)'].sum() if 'Deposit Amount(INR)' in df.columns else 0,
    }
    
    # Check for Category column (case-insensitive)
    category_col = None
    for col in df.columns:
        if col.lower() == 'category':
            category_col = col
            break
    
    if category_col:
        stats['categories_found'] = df[category_col].nunique()
        stats['category_breakdown'] = df[category_col].value_counts().to_dict()
    
    return stats


def print_summary_stats(stats: dict) -> None:
    """
    Print summary statistics in a formatted way.
    
    Args:
        stats: Dictionary with summary statistics
    """
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total Transactions: {stats['total_transactions']}")
    print(f"Total Withdrawals: ₹{stats['total_withdrawals']:,.2f}")
    print(f"Total Deposits: ₹{stats['total_deposits']:,.2f}")
    
    if 'categories_found' in stats:
        print(f"\nCategories Found: {stats['categories_found']}")
        if 'category_breakdown' in stats and stats['category_breakdown']:
            print("\nCategory Breakdown:")
            for category, count in stats['category_breakdown'].items():
                print(f"  - {category}: {count}")
    
    print("="*50 + "\n")

