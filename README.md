# 💎 Diamond Price Prediction: Machine Learning Project 💻📊

Welcome to the **Diamond Price Prediction** project, developed as part of the **Machine Learning using Python** course (3155202). This repository hosts our end-semester project which aims to build a robust machine learning model to automate the valuation of diamonds, addressing inefficiencies in traditional pricing methods.

## 🎯 Project Overview

This project aims to develop a scalable and consistent pricing system using advanced machine learning models. By automating the valuation process, we seek to revolutionize the diamond industry, which is traditionally dependent on subjective and time-intensive manual appraisals.

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Dataset](#dataset)
5. [Methodology](#methodology)
6. [Results and Insights](#results-and-insights)
7. [Web Application Features](#web-application-features)
8. [Future Enhancements](#future-enhancements)
9. [Usage](#usage)
10. [Project Structure](#project-structure)
11. [Contributing](#contributing)
12. [License](#license)
13. [Acknowledgment](#acknowledgment)
14. [References](#references)

## Introduction

In the diamond industry, pricing traditionally relies on human expertise, leading to inconsistencies and inefficiencies. This project leverages machine learning to provide a data-driven solution, ensuring consistent and objective pricing.

## 🎯 Problem Statement

Gem Stones Co. Ltd. faces several challenges in optimizing their profit margins:

1. 🚫 **Subjective Valuation**: Reliance on human expertise leads to inconsistencies.
2. 🕒 **Time-Intensive Appraisals**: Manual processes slow down operations.
3. 📈 **Limited Scalability**: Current methods struggle to scale with demand.
4. ⚡ **Real-Time Adjustments**: Inability to promptly reflect market changes.

## Objectives

### 🎯 Primary Objectives

- Develop a high-accuracy machine learning model with:
  - R² > 0.95
  - RMSE < 5% of mean price
  - MAE < $500
- Establish an automated system with:
  - Low latency (< 100ms)
  - High availability (> 99.9%)

### Secondary Objectives

- Identify key pricing factors.
- Provide actionable insights for better inventory management.
- Support data-driven decisions in trading.

## 📊 Dataset

We used the **[Gemstone Price Prediction Dataset](https://www.kaggle.com/datasets/colearninglounge/gemstone-price-prediction)** from Kaggle. It contains approximately 27,000 records of cubic zirconia attributes, providing a comprehensive base for training and testing our models.

### Dataset Highlights:

- 📦 **193500+ Records**
- 🏷️ **9 Features**: Including carat weight, cut, color, clarity, and dimensions.
- 📊 **High-Quality Data**: No missing values or duplicates, ensuring robust analysis.

## Methodology

### Data Preprocessing

- 🧹 **Numerical Features**: Median imputation.
- 🏷️ **Categorical Features**: Ordinal encoding for `cut`, `color`, and `clarity`.

### Data Pipeline Architecture 🛠️

```python
# Example of our complete preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num_pipeline', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numerical_features),
    ('cat_pipeline', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories]))
    ]), categorical_features)
])
```

### Model Development 🤖

- **Linear Models**: Linear Regression, Lasso, Ridge, Elastic Net.
- **Tree-based Models**: Decision Tree, Random Forest, XGBoost, CatBoost.
- **Neural Networks**: Fully connected models with ReLU activation.

### Training Strategy

- 🌀 **k-Fold Cross-Validation**
- 🔍 **Hyperparameter Tuning** using Grid and Randomized Search.

## 📈 Results and Insights

Our models achieved the following performance metrics:

| Model           | R² Score | RMSE ($) | MAE ($) | Training Time (s) |
|-----------------|----------|----------|---------|-------------------|
| Linear Regression | 0.9363 | 1014.63  | 675.08  | 0.15              |
| Lasso            | 0.9368   | 1014.61  | 675.27  | 0.22              |
| Ridge            | 0.9367   | 1014.64  | 675.22  | 0.18              |
| Elastic Net      | 0.8917   | 1327.12  | 1002.73 | 0.25              |
| Decision Tree    | 0.9704   | 854.98   | 427.16  | 0.35              |
| Random Forest    | 0.9768   | 611.83   | 309.79  | 90.2              |
| XGBoost          | 0.9796   | 587.89   | 297.25  | 10.85             |
| CatBoost         | 0.9783   | 701.00   | 445.00  | 35.95             |
| Neural Network   | 0.9745   | 642.24   | 343.34  | 150.32            |

### Key Insights:

- **Best Model**: XGBoost with an impressive R² of 0.98.
- **Feature Importance**:
  - Carat Weight: 45%
  - Cut Quality: 15%
  - Clarity: 12%
  - Color: 10%

### Business Impact

- **Reduced Operational Costs**
- **Improved Pricing Accuracy**
- **Enhanced Customer Satisfaction**

## Web Application Features 🌐

The Streamlit web application provides:

- 💎 Interactive diamond price prediction
- 📊 Real-time visualization of predictions
- 📈 Feature importance analysis
- 🔄 Batch prediction capabilities
- 📱 Mobile-responsive design

## 🔮 Future Enhancements

1. **Advanced Models** 🚀
   - Deep learning implementation.
   - Ensemble methods.
   - Time series analysis.

2. **System Improvements** 💫
   - Real-time pricing.
   - A/B testing framework.
   - Automated retraining.
   - Mobile Integration
   - Batch Processing to handle high-volume pricing efficiently.

## 🚀 Quick Start (Usage)

```bash
# Clone the repository
git clone https://github.com/pruthakjani5/diamond-price-prediction.git

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
```

## 📚 Project Structure

```
diamond-price-prediction/
│
├── 📁 data/                  # Dataset files
├── 📁 notebooks/             # Jupyter notebooks for analysis
├── 📁 models/                # Trained models
├── 📄 app.py                 # Streamlit application
├── 📄 requirements.txt       # Dependencies
└── 📄 README.md              # Project documentation
```

## 👥 Contributing

Contributions are welcome! Feel free to:

1. 🍴 Fork the repository.
2. 🔧 Create a feature branch.
3. 💡 Submit a pull request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgment

This project was developed by **Pruthak Jani** as part of the **Machine Learning using Python** course at LDCE. Special thanks to the subject faculty for their guidance and support.

## 📚 References

1. 📊 [Kaggle Dataset](https://www.kaggle.com/datasets/colearninglounge/gemstone-price-prediction)
2. 📘 [XGBoost Documentation](https://xgboost.readthedocs.io/)
3. 📱 [Streamlit Documentation](https://docs.streamlit.io/)
4. 🔬 [Scikit-learn Documentation](https://scikit-learn.org/)

---

Let's innovate together and transform the diamond industry! 🚀
