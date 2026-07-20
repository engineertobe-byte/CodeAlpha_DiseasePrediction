import os
import pandas as pd
from src.preprocessing import DataPreprocessor
from src.training import ModelTrainer
from src.visualization import DataVisualizer

def main():
    print("🚀 Starting Disease Prediction Project (Task 4)...")
    
    # 1. Load Data
    data_path = 'data/heart_disease_uci.csv'
    if not os.path.exists(data_path):
        print(f"❌ Error: File {data_path} not found. Please download it.")
        return

    df = pd.read_csv(data_path)
    print("✅ Data loaded successfully.")

    # 2. Preprocessing
    preprocessor = DataPreprocessor(df)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    print("✅ Preprocessing completed.")

    # 3. Visualization (EDA)
    visualizer = DataVisualizer(preprocessor.df_cleaned)
    visualizer.plot_correlation_matrix()
    visualizer.plot_target_distribution()
    print("✅ Visualizations generated in 'plots/' folder.")

    # 4. Training and Evaluation
    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    best_model, metrics = trainer.train_and_evaluate()
    
    print("\n📊 Best Model Results:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
        
    print("\n🎉 Project completed successfully! Check plots and saved model.")

if __name__ == "__main__":
    main()
