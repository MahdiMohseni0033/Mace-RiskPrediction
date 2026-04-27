import pandas as pd
import sys

input_file = 'datasets/merged_proteomics_base.csv'
output_file = 'dataset_profile.txt'

def profile_dataset(csv_path, out_path):
    try:
        print(f"Loading '{csv_path}' for profiling...")
        df = pd.read_csv(csv_path)
        
        num_rows, num_cols = df.shape
        
        # Calculate missing percentage for columns
        col_missing = (df.isnull().sum() / num_rows) * 100
        col_missing_sorted = col_missing.sort_values(ascending=False)
        
        # Calculate missing percentage for rows
        row_missing = (df.isnull().sum(axis=1) / num_cols) * 100
        
        with open(out_path, 'w') as f:
            f.write("===============================\n")
            f.write("        DATASET PROFILE        \n")
            f.write("===============================\n\n")
            f.write(f"Total Rows: {num_rows}\n")
            f.write(f"Total Columns: {num_cols}\n\n")
            
            f.write("----------------------------------------------\n")
            f.write(" Missing Values by Column (%) - Descending \n")
            f.write("----------------------------------------------\n")
            for col, pct in col_missing_sorted.items():
                f.write(f"{col}: {pct:.2f}%\n")
                
            f.write("\n----------------------------------------------\n")
            f.write(" Missing Values by Row (%) \n")
            f.write("----------------------------------------------\n")
            for idx, pct in row_missing.items():
                # Provide row index (0-based) and percentage
                f.write(f"Row {idx}: {pct:.2f}%\n")
                
        print(f"Profiling complete. Results saved to '{out_path}'")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    profile_dataset(input_file, output_file)
