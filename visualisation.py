import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataVisualizer:
    def __init__(self, df):
        self.df = df
        os.makedirs('plots', exist_ok=True)

    def plot_correlation_matrix(self):
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix - Disease Prediction")
        plt.savefig('plots/correlation_matrix.png')
        plt.close()

    def plot_target_distribution(self):
        plt.figure(figsize=(8, 6))
        # Adapt target column name based on your dataset
        target_col = 'num' if 'num' in self.df.columns else self.df.columns[-1]
        sns.countplot(x=self.df[target_col].apply(lambda x: 1 if x > 0 else 0))
        plt.title("Target Distribution (Healthy vs Diseased)")
        plt.xlabel("Status (0: Healthy, 1: Diseased)")
        plt.ylabel("Patient Count")
        plt.savefig('plots/target_distribution.png')
        plt.close()
