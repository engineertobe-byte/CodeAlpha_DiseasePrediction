import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, df):
        self.df = df
        self.df_cleaned = None

    def prepare_data(self):
        # Basic cleaning: drop rows with missing values
        self.df_cleaned = self.df.dropna()
        
        # Separate features and target
        # Note: Adapt 'num' to your specific CSV column name if different
        target_column = 'num' 
        if target_column not in self.df_cleaned.columns:
            target_column = [col for col in self.df_cleaned.columns if col.lower() in ['target', 'disease', 'output']][0]

        X = self.df_cleaned.drop(columns=[target_column])
        y = self.df_cleaned[target_column]

        # Convert target to binary (0 = healthy, 1+ = diseased)
        y = y.apply(lambda x: 1 if x > 0 else 0)

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test
