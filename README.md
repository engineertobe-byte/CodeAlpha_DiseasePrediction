# 🩺 Disease Prediction Model | CodeAlpha Internship Task 4

## 📌 Project Overview
This project predicts disease likelihood using structured medical data. Leveraging Python and Scikit-learn, it implements Logistic Regression, Random Forest, and Gradient Boosting. The model is optimized for high Precision, Recall, and F1-Score to ensure reliable diagnostic support in healthcare applications.

## 🛠️ Technologies Used
*   **Language:** Python
*   **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib
*   **Algorithms:** Logistic Regression, Random Forest Classifier, Gradient Boosting Classifier

## 📂 Project Structure
```text
CodeAlpha_DiseasePrediction/
│
├── data/                  # Dataset storage (Heart Disease UCI)
├── models/                # Saved trained models (.pkl)
├── plots/                 # Generated EDA visualizations
├── src/                   # Source code modules
│   ├── preprocessing.py   # Data cleaning and scaling
│   ├── training.py        # Model training and evaluation
│   └── visualization.py   # Data plotting functions
│
├── main.py                # Main execution script
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/CodeAlpha_DiseasePrediction.git
    cd CodeAlpha_DiseasePrediction
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download Dataset:**
    Place the `heart_disease_uci.csv` file inside the `data/` folder. You can find it on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/).
4.  **Run the application:**
    ```bash
    python main.py
    ```

## 📊 Model Evaluation Metrics
The project evaluates performance using the following key metrics:
*   **Precision:** Accuracy of positive predictions.
*   **Recall:** Ability to find all positive instances.
*   **F1-Score:** Harmonic mean of Precision and Recall.
*   **ROC-AUC:** Measure of separability between classes.

## 📝 License
This project is part of the CodeAlpha Machine Learning Internship program.
