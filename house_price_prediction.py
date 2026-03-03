"""
House Price Prediction System - Skillario Project
Advanced Regression Techniques | Single-File Implementation

Complete pipeline: Load → EDA → Preprocess → Feature Engineering → Train → Evaluate
Dataset: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
TARGET_COLUMN = "SalePrice"
RANDOM_STATE = 42
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING & EDA
# =============================================================================
def load_data():
    """Load train and test datasets from CSV files."""
    train_path = RAW_DATA_DIR / TRAIN_FILE
    test_path = RAW_DATA_DIR / TEST_FILE
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Download from: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data"
        )
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path.exists() else None
    return train_df, test_df


def plot_missing_values(df, save_path=None):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(8, len(missing) * 0.3)))
    missing.plot(kind='barh', ax=ax, color='coral', alpha=0.8)
    ax.set_xlabel('Number of Missing Values')
    ax.set_title('Missing Values by Feature')
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_target_distribution(df, target=TARGET_COLUMN, save_path=None):
    if target not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df[target].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Sale Price ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Sale Prices')
    axes[1].hist(np.log1p(df[target].dropna()), bins=50, color='seagreen', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Log(Sale Price)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Log-Transformed Sale Prices')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_correlation_heatmap(df, top_n=20, save_path=None):
    numeric_df = df.select_dtypes(include=[np.number])
    if TARGET_COLUMN not in numeric_df.columns:
        return
    correlations = numeric_df.corr()[TARGET_COLUMN].abs().sort_values(ascending=False)
    top_features = correlations.head(top_n + 1).index.tolist()
    corr_matrix = numeric_df[top_features].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title(f'Top {top_n} Features Correlation with Sale Price')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_eda_report(train_df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        'missing_values': train_df.isnull().sum(),
        'missing_percent': (train_df.isnull().sum() / len(train_df) * 100).round(2),
    }
    missing_df = pd.DataFrame({'Missing_Count': summary['missing_values'], 'Missing_Percent': summary['missing_percent']})
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    if not missing_df.empty:
        missing_df.to_csv(OUTPUT_DIR / 'missing_values_report.csv')
    plot_missing_values(train_df, OUTPUT_DIR / 'missing_values.png')
    plot_target_distribution(train_df, save_path=OUTPUT_DIR / 'target_distribution.png')
    plot_correlation_heatmap(train_df, top_n=15, save_path=OUTPUT_DIR / 'correlation_heatmap.png')
    print("EDA report generated in", OUTPUT_DIR)


# =============================================================================
# PREPROCESSING
# =============================================================================
def handle_missing_values(df, is_train=True):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COLUMN in numeric_cols and not is_train:
        numeric_cols.remove(TARGET_COLUMN)
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            df[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown', inplace=True)
    return df


def remove_outliers(df, target=TARGET_COLUMN, threshold=1.5):
    if target not in df.columns:
        return df
    df = df.copy()
    Q1, Q3 = df[target].quantile(0.25), df[target].quantile(0.75)
    IQR = Q3 - Q1
    lb, ub = Q1 - threshold * IQR, Q3 + threshold * IQR
    return df[(df[target] >= lb) & (df[target] <= ub)]


def encode_categorical(df, label_encoders=None, fit=True):
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    label_encoders = label_encoders or {}
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        if col not in label_encoders:
            label_encoders[col] = LabelEncoder()
            if fit:
                label_encoders[col].fit(df[col].unique())
        def safe_transform(x):
            return label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1
        df[col] = df[col].apply(safe_transform)
    return df, label_encoders


def preprocess_pipeline(train_df, test_df=None, remove_outliers_flag=True):
    train_processed = handle_missing_values(train_df, is_train=True)
    test_processed = handle_missing_values(test_df, is_train=False) if test_df is not None else None
    if remove_outliers_flag and TARGET_COLUMN in train_processed.columns:
        initial_len = len(train_processed)
        train_processed = remove_outliers(train_processed)
        print(f"Removed {initial_len - len(train_processed)} outliers from training data")
    train_processed, encoders = encode_categorical(train_processed, fit=True)
    if test_processed is not None:
        test_processed, _ = encode_categorical(test_processed, label_encoders=encoders, fit=False)
        feature_cols = [c for c in train_processed.columns if c != TARGET_COLUMN]
        for col in feature_cols:
            if col not in test_processed.columns:
                test_processed[col] = 0
        test_processed = test_processed[feature_cols]
    return train_processed, test_processed, encoders


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def create_derived_features(df):
    df = df.copy()
    if all(c in df.columns for c in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    if all(c in df.columns for c in bath_cols):
        df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    if 'YearBuilt' in df.columns:
        ref_year = df['YrSold'].max() if 'YrSold' in df.columns else 2010
        df['HouseAge'] = ref_year - df['YearBuilt']
    if all(c in df.columns for c in ['YearBuilt', 'YearRemodAdd']):
        df['YearsSinceRemod'] = df['YearRemodAdd'] - df['YearBuilt']
    porch_cols = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    existing_porch = [c for c in porch_cols if c in df.columns]
    if existing_porch:
        df['TotalPorchSF'] = df[existing_porch].sum(axis=1)
    if all(c in df.columns for c in ['OverallQual', 'GrLivArea']):
        df['Qual_x_Area'] = df['OverallQual'] * df['GrLivArea']
    if all(c in df.columns for c in ['OverallQual', 'TotalBsmtSF']):
        df['Qual_x_Bsmt'] = df['OverallQual'] * df['TotalBsmtSF']
    if all(c in df.columns for c in ['GarageCars', 'GarageArea']):
        df['GarageScore'] = df['GarageCars'] * df['GarageArea']
    if 'PoolArea' in df.columns:
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    if 'GarageArea' in df.columns:
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    if 'Fireplaces' in df.columns:
        df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    return df


def select_features(df, target=TARGET_COLUMN):
    exclude = [target, 'Id']
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.int64, np.float64, np.int32, np.float32]]


def apply_feature_engineering(train_df, test_df=None):
    train_fe = create_derived_features(train_df)
    test_fe = create_derived_features(test_df) if test_df is not None else None
    return train_fe, test_fe


# =============================================================================
# MODELS
# =============================================================================
def get_models():
    return {
        'Ridge': Ridge(alpha=10.0, random_state=RANDOM_STATE),
        'Lasso': Lasso(alpha=0.0005, random_state=RANDOM_STATE),
        'ElasticNet': ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=RANDOM_STATE),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=RANDOM_STATE),
    }


def train_and_evaluate(X_train, y_train, cv_folds=5, use_log_target=True):
    y_train = np.log1p(y_train) if use_log_target else y_train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    models = get_models()
    scaled_models = ['Ridge', 'Lasso', 'ElasticNet']
    results = {}
    for name, model in models.items():
        X = X_scaled if name in scaled_models else X_train.values
        scores = cross_val_score(model, X, y_train, cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1)
        rmse_scores = np.sqrt(-scores)
        results[name] = {'mean_rmse': rmse_scores.mean(), 'std_rmse': rmse_scores.std(), 'scores': rmse_scores}
        print(f"{name}: RMSE = {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")
    return results


def train_best_model(X_train, y_train, model_name='Gradient Boosting', use_log_target=True):
    y_train = np.log1p(y_train) if use_log_target else y_train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    models = get_models()
    model = models.get(model_name, models['Gradient Boosting'])
    scaled_models = ['Ridge', 'Lasso', 'ElasticNet']
    X = X_scaled if model_name in scaled_models else X_train.values
    model.fit(X, y_train)
    return model, scaler if model_name in scaled_models else None


def predict_with_model(model, scaler, X, model_name):
    scaled_models = ['Ridge', 'Lasso', 'ElasticNet']
    X_processed = scaler.transform(X) if model_name in scaled_models and scaler else X.values
    return np.expm1(model.predict(X_processed))


# =============================================================================
# EVALUATION
# =============================================================================
def calculate_metrics(y_true, y_pred, use_log=True):
    if use_log:
        rmse = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
        metric_name = 'RMSLE'
    else:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        metric_name = 'RMSE'
    return {metric_name: rmse, 'MAE': mean_absolute_error(y_true, y_pred), 'R2': r2_score(y_true, y_pred)}


def plot_model_comparison(results, save_path=None):
    model_names = list(results.keys())
    mean_rmse = [results[m]['mean_rmse'] for m in model_names]
    std_rmse = [results[m]['std_rmse'] for m in model_names]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
    bars = ax.bar(model_names, mean_rmse, yerr=std_rmse, capsize=5, color=colors, edgecolor='black')
    ax.set_ylabel('RMSE (Cross-Validation)')
    ax.set_title('Model Comparison - House Price Prediction')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    for bar, val in zip(bars, mean_rmse):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_predictions_vs_actual(y_true, y_pred, model_name="Model", save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_true, y_pred, alpha=0.5, c='steelblue', edgecolors='black', s=20)
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Sale Price ($)')
    axes[0].set_ylabel('Predicted Sale Price ($)')
    axes[0].set_title(f'{model_name}: Predicted vs Actual')
    axes[0].legend()
    axes[0].set_xlim(0, max_val * 1.05)
    axes[0].set_ylim(0, max_val * 1.05)
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='red', linestyle='--', lw=2)
    axes[1].set_xlabel('Residual (Actual - Predicted)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        ax.barh(range(len(indices)), importance[indices], color='teal', alpha=0.8)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top Feature Importances')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def generate_evaluation_report(results, y_true, y_pred, model_name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_model_comparison(results, OUTPUT_DIR / 'model_comparison.png')
    plot_predictions_vs_actual(y_true, y_pred, model_name, OUTPUT_DIR / 'predictions_vs_actual.png')
    metrics = calculate_metrics(y_true, y_pred)
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / 'evaluation_metrics.csv', index=False)
    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_pipeline():
    print("=" * 60)
    print("  HOUSE PRICE PREDICTION SYSTEM - Skillario Project")
    print("  Advanced Regression Techniques")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    train_df, test_df = load_data()
    print(f"  Training samples: {len(train_df)}, Features: {len(train_df.columns)}")
    if test_df is not None:
        print(f"  Test samples: {len(test_df)}")

    print("\n[2/6] Exploratory Data Analysis...")
    generate_eda_report(train_df)

    print("\n[3/6] Feature Engineering...")
    train_df, test_df = apply_feature_engineering(train_df, test_df)

    print("\n[4/6] Data Preprocessing...")
    train_processed, test_processed, encoders = preprocess_pipeline(train_df, test_df, remove_outliers_flag=True)

    feature_cols = select_features(train_processed)
    X = train_processed[feature_cols]
    y = train_processed[TARGET_COLUMN]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    print("\n[5/6] Training and Evaluating Models...")
    cv_results = train_and_evaluate(X_train, y_train, cv_folds=5)
    best_model_name = min(cv_results.keys(), key=lambda k: cv_results[k]['mean_rmse'])
    print(f"\n  Best Model: {best_model_name} (RMSE: {cv_results[best_model_name]['mean_rmse']:.4f})")

    model, scaler = train_best_model(X_train, y_train, model_name=best_model_name)
    y_pred = predict_with_model(model, scaler, X_val, best_model_name)
    generate_evaluation_report(cv_results, y_val.values, y_pred, best_model_name)

    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model, feature_cols, top_n=20, save_path=OUTPUT_DIR / 'feature_importance.png')

    print("\n[6/6] Saving outputs...")
    train_processed.to_csv(PROCESSED_DATA_DIR / 'train_processed.csv', index=False)
    if test_processed is not None:
        test_processed.to_csv(PROCESSED_DATA_DIR / 'test_processed.csv', index=False)

    if test_processed is not None and test_df is not None and 'Id' in test_df.columns:
        X_test = test_processed[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        test_predictions = predict_with_model(model, scaler, X_test, best_model_name)
        pd.DataFrame({'Id': test_df['Id'], 'SalePrice': test_predictions}).to_csv(OUTPUT_DIR / 'submission.csv', index=False)
        print(f"  Submission saved: {OUTPUT_DIR / 'submission.csv'}")

    metrics = calculate_metrics(y_val.values, y_pred)
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE - Final Validation Metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")
    print("=" * 60)
    return model, scaler, cv_results, metrics


if __name__ == "__main__":
    run_pipeline()
