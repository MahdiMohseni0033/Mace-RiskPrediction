import pandas as pd
from sklearn.impute import KNNImputer
import sys

input_file = 'datasets/merged_proteomics_base.csv'
output_file = 'datasets/imputed_proteomics_base.csv'

try:
    print(f"Loading '{input_file}'...")
    df = pd.read_csv(input_file)
    
    # Check total missing before
    missing_before = df.isnull().sum().sum()
    print(f"Total missing values before imputation: {missing_before}")

    # We need to exclude the ID column from the mathematical imputation calculations
    id_col = 'EuroSCOREPatient ID'
    ids = df[id_col]
    features = df.drop(columns=[id_col])
    
    # KNN Imputer looks at the most similar rows to estimate the missing value
    print("Applying KNN Imputer (k=5)...")
    imputer = KNNImputer(n_neighbors=5)
    imputed_features = imputer.fit_transform(features)
    
    # Reconstruct the DataFrame with the exact same columns
    imputed_df = pd.DataFrame(imputed_features, columns=features.columns)
    
    # Put the ID column back in the front
    imputed_df.insert(0, id_col, ids)
    
    # Verify imputation
    missing_after = imputed_df.isnull().sum().sum()
    print(f"Total missing values after imputation: {missing_after}")
    
    # Save as independent dataset
    imputed_df.to_csv(output_file, index=False)
    print(f"Saved new, completed dataset to '{output_file}'")

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
