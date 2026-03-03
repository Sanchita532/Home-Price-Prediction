# Home-Price-Prediction
# 🏠 House Price Prediction System

**Skillario Project** | Advanced Regression Techniques

A comprehensive machine learning solution for predicting house prices using property-related features. Built with Python, Pandas, Matplotlib, and Scikit-learn.

---

## 📋 Project Overview

| Aspect | Description |
|--------|-------------|
| **Problem** | Accurately estimating house prices is challenging due to multiple influencing factors |
| **Objective** | Predict house prices using property-related features |
| **Dataset** | [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) |
| **Approach** | Regression, Feature Engineering, Model Evaluation |

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd house_price_prediction
pip install -r requirements.txt
```

### 2. Download Dataset

1. Create a [Kaggle](https://www.kaggle.com) account
2. Go to the [competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
3. Click **Download** to get the dataset
4. Extract and place `train.csv` and `test.csv` in `data/raw/` folder

**Alternative (Kaggle API):**
```bash
kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip -d data/raw/
```

### 3. Run the Pipeline

```bash
python main.py
```

---

## 📁 Project Structure

```
house_price_prediction/
├── config.py              # Configuration & paths
├── data_loader.py         # Data loading & EDA
├── preprocessing.py      # Cleaning & encoding
├── feature_engineering.py # Derived features
├── models.py              # Regression models
├── evaluation.py          # Metrics & visualization
├── main.py                # Main pipeline
├── requirements.txt       # Dependencies
├── README.md
├── data/
│   ├── raw/               # Place train.csv, test.csv here
│   └── processed/         # Cleaned data (auto-generated)
├── output/                # Plots, reports, submission (auto-generated)
└── models/                # Saved models (auto-generated)
```

---

## 🔧 Pipeline Components

### 1. **Data Loading & Exploration**
- Load train/test datasets
- Missing value analysis
- Target distribution visualization
- Correlation heatmap

### 2. **Data Preprocessing**
- Missing value imputation (median for numeric, mode for categorical)
- Outlier removal (IQR method)
- Label encoding for categorical variables

### 3. **Feature Engineering**
- `TotalSF` - Total square footage
- `TotalBath` - Total bathrooms
- `HouseAge` - Age of property
- `TotalPorchSF` - Total porch area
- `Qual_x_Area` - Quality × Living area interaction
- `HasPool`, `HasGarage`, `HasFireplace` - Binary features

### 4. **Models Trained**
- **Ridge Regression** - L2 regularization
- **Lasso Regression** - L1 regularization  
- **ElasticNet** - Combined L1 + L2
- **Random Forest** - Ensemble of decision trees
- **Gradient Boosting** - Sequential ensemble

### 5. **Evaluation**
- 5-fold cross-validation
- RMSE (Root Mean Squared Error)
- RMSLE (Log-transformed, Kaggle metric)
- R² score
- Predicted vs Actual plots
- Feature importance analysis

---

## 📊 Outputs Generated

| File | Description |
|------|-------------|
| `output/missing_values.png` | Missing data visualization |
| `output/target_distribution.png` | Sale price distribution |
| `output/correlation_heatmap.png` | Feature correlations |
| `output/model_comparison.png` | Model performance comparison |
| `output/predictions_vs_actual.png` | Prediction accuracy |
| `output/feature_importance.png` | Top predictive features |
| `output/submission.csv` | Kaggle submission format |

---

## 🛠️ Tools & Libraries

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Matplotlib** - Visualization
- **Seaborn** - Statistical plots
- **Scikit-learn** - ML models & preprocessing

---

## 📈 Expected Results

The pipeline typically achieves:
- **RMSLE**: ~0.14-0.18 (varies with hyperparameters)
- **R²**: 0.85-0.92 on validation set
- **Best Model**: Usually Gradient Boosting or Random Forest

---

## 👤 Author

Developed as part of the **Skillario** hiring project — demonstrating hands-on experience with structured data prediction problems.

---

## 📄 License

This project is for educational and portfolio purposes.
