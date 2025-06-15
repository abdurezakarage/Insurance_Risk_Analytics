import pandas as pd
import numpy as np
from pathlib import Path
class DataProccessor:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
#process the data
    def process(self):
       
        df = pd.read_csv(self.input_file, sep='|', encoding='utf-8', low_memory=False)
        
        # Basic data cleaning
        required_columns = ['Province', 'PostalCode', 'Gender', 'TotalPremium', 'TotalClaims']
        df = df.dropna(subset=required_columns)

        # Create binary claim indicator
        df['has_claim'] = (df['TotalClaims'] > 0).astype(int)

        # Ensure numeric columns are float
        numeric_columns = ['TotalPremium', 'TotalClaims']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Create processed data directory 
        output_path = Path(self.output_file).parent
        output_path.mkdir(parents=True, exist_ok=True)

        # Save processed data
        df.to_csv(self.output_file, index=False)
        print(f"Processed data saved to {self.output_file}")
        
        # basic statistics
        print("\nBasic Statistics:")
        print(f"Total number of records: {len(df)}")
        print(f"Number of claims: {df['has_claim'].sum()}")
        print(f"Average TotalClaims: {df['TotalClaims'].mean():.2f}")
        print(f"Average TotalPremium: {df['TotalPremium'].mean():.2f}")
        
        # unique values for categorical columns
        print("\nUnique values in categorical columns:")
        print("\nProvinces:", df['Province'].unique())
        print("\nGender values:", df['Gender'].unique())
        print("\nNumber of unique postal codes:", df['PostalCode'].nunique()) 

        return df
    
