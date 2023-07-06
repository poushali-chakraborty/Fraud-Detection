# Fraud-Detection
📂 Fraud Detection using Decision Trees

This project aims to detect fraud using a Decision Tree classifier. The code is implemented in a Jupyter Notebook and leverages various Python libraries including pandas, NumPy, Matplotlib, seaborn, scikit-learn, and xgboost.

🔍 Dataset

The financial transaction dataset, named `fraud dataset.csv`, is used for fraud detection. It contains information about different types of transactions, including fraudulent ones. The dataset is loaded into a pandas DataFrame for further processing.

📝 Code Overview

1️⃣ Data Preprocessing:
   - Renaming columns to improve readability
   - Handling missing values
   - Converting data types to appropriate formats
   - Creating additional categorical features based on existing data
   
2️⃣ Exploratory Data Analysis (EDA):
   - Statistical summary of the dataset
   - Identifying missing values
   - Checking data formats
   
3️⃣ Visualization:
   - Count plots showing the number of transactions per type, highlighting fraud cases
   - Histogram plots displaying the distribution of transaction amounts, with fraud cases marked
   
4️⃣ Feature Engineering:
   - Dropping unnecessary columns
   - Applying logarithmic transformation to selected columns
   
5️⃣ Model Training and Evaluation:
   - Splitting the dataset into training and testing sets
   - Training a Decision Tree classifier using the training set
   - Evaluating the model's accuracy score on the testing set
   - Visualizing the ROC AUC score, confusion matrix, and F1-score of the model
   
6️⃣ Predictions and Feature Importances:
   - Creating predictions using the trained model on a sample dataset
   - Visualizing the feature importances using the Decision Tree model

🔧 Dependencies

To run the code, make sure you have the following libraries installed:
- pandas
- NumPy
- Matplotlib
- seaborn
- scikit-learn
- xgboost

💡 How to Use

1. Install the required dependencies using `pip install pandas numpy matplotlib seaborn scikit-learn xgboost`.

2. Download the Jupyter Notebook (`fraud_detection.ipynb`) and the financial transaction dataset (`fraud dataset.csv`).

3. Open the Jupyter Notebook in a Jupyter Notebook environment.

4. Run each cell in the notebook sequentially to execute the code step-by-step.

5. Observe the EDA visualizations and the performance of the Decision Tree classifier.


🙏 Acknowledgments

The financial transaction dataset used in this project is publicly available on Kaggle. The code in this repository is inspired by various machine learning and data analysis techniques from scikit-learn, xgboost, and other libraries.

📚 References

1. [Financial Transaction Dataset on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1?select=PS_20174392719_1491204439457_log.csv)
2. [pandas Documentation](https://pandas.pydata.org/docs/)
3. [NumPy Documentation](https://numpy.org/doc/)
4. [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
5. [seaborn Documentation](https://seaborn.pydata.org/api.html)
6. [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
7. [xgboost Documentation](https://xgboost.readthedocs.io/en/latest/index.html)
