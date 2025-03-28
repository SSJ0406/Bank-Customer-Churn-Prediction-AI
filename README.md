# 💳 Bank Customer Churn Prediction

## 📋 Overview
This project uses machine learning to predict customer churn in a banking context, based on demographic and behavioral data from a Kaggle dataset. Key challenges included:

- **Class Imbalance**: Dataset skewed heavily towards non-churned customers
- **Outlier Detection**: Identification and removal of extreme data points
- **Feature Engineering**: Processing categorical (country, gender) and numerical features (credit score, age, balance)
- **Model Selection and Tuning**: Evaluating various machine learning models

## 🎯 Key Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | 86.83% | 78.64% | 45.77% | 57.86% | 0.8591 |
| **Random Forest** | 86.05% | 78.17% | - | - | 0.8575 |

## 🛠️ Implementation Details

### Data Preprocessing
```python
# Outlier Removal
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['age'] < (Q1 - 1.5 * IQR)) | (df['age'] > (Q3 + 1.5 * IQR)))]

# Feature Scaling and Encoding
scaler = StandardScaler()
encoder = OneHotEncoder()

# Splitting Features and Labels
X = df.drop(columns=['churn'])
y = df['churn']
```

### Addressing Class Imbalance
```python
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=y
)
```

### Models Implemented

```python
# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)

# Support Vector Machine (SVM)
svm_model = SVC(probability=True, random_state=42)

# Random Forest Classifier
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)

# XGBoost Classifier
xgb_model = XGBClassifier(scale_pos_weight=class_weights[1]/class_weights[0], random_state=42)
```

### Cross-Validation
```python
cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='roc_auc')
print(f"Average ROC-AUC: {cv_scores.mean():.4f}")
```

## 📊 Data Analysis Approaches

- **Exploratory Data Analysis**: Histograms, boxplots, and distribution analysis
- **Feature Correlation**: Explored relationships between variables
- **Class Imbalance Visualization**: Analyzed target variable distribution
- **Model Performance Evaluation**: Compared metrics across multiple algorithms

## 🔧 Technologies Used

- **Python**: Primary programming language
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Model implementation and evaluation
- **XGBoost**: Gradient boosting implementation
- **Matplotlib & Seaborn**: Data visualization

## 🗂️ Repository Structure

```
├── Bank Customer Churn Prediction.csv   # Raw dataset
├── Bank Customer Churn Prediction.ipynb # Analysis notebook
├── README.md                            # Project documentation
```

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/SSJ0406/Bank-Customer-Churn-Prediction-AI.git
cd Bank-Customer-Churn-Prediction-AI
```

2. Install required dependencies:
```bash
pip install scikit-learn pandas numpy xgboost matplotlib seaborn
```

3. Run the Jupyter notebook:
```bash
jupyter notebook "Bank Customer Churn Prediction.ipynb"
```

## 💡 Business Impact & Future Work

- **Business Relevance**:
  - Enables banks to identify at-risk customers for targeted retention strategies
  - Provides insights into factors contributing to customer churn

- **Future Directions**:
  - Explore deep learning approaches for improved prediction accuracy
  - Implement real-time model deployment for ongoing monitoring
  - Incorporate additional customer interaction data for enhanced predictions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Kaggle for providing the dataset and community support
