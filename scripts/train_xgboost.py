"""
XGBoost Risk Classifier
Classifies DeFi protocols into risk categories: low/medium/high
Uses same feature engineering pipeline as LSTM
"""

import os
import sys
import logging
import pickle
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import xgboost as xgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.feature_engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_risk_label(apy: float, tvl: float, utilization: float) -> str:
    """
    Calculate risk label based on protocol characteristics
    
    Risk Criteria (adjusted for better balance):
    - Low: Stable APY (<5%), High TVL (>$500K), Low util (<70%)
    - High: High APY (>20%), Low TVL (<$50K), or Very High util (>85%)
    - Medium: Everything else
    
    Args:
        apy: Current APY percentage
        tvl: Total Value Locked in USD
        utilization: Utilization ratio (0-100)
        
    Returns:
        Risk label: 'low', 'medium', or 'high'
    """
    risk_score = 0
    
    # APY-based risk (adjusted thresholds)
    if apy > 20:
        risk_score += 2
    elif apy > 8:
        risk_score += 1
    elif apy < 5:
        risk_score -= 1  # Lower risk
    
    # TVL-based risk (inverse relationship, adjusted thresholds)
    if tvl < 50_000:
        risk_score += 2
    elif tvl < 500_000:
        risk_score += 1
    elif tvl > 1_000_000:
        risk_score -= 1  # Lower risk
    
    # Utilization-based risk (adjusted thresholds)
    if utilization > 85:
        risk_score += 2
    elif utilization > 75:
        risk_score += 1
    
    # Map score to label (adjusted thresholds)
    if risk_score >= 3:
        return 'high'
    elif risk_score <= 0:
        return 'low'
    else:
        return 'medium'


def prepare_risk_data(db_url: str):
    """
    Prepare training data for risk classification
    
    Returns:
        X: Feature matrix
        y: Risk labels
        assets: Asset names
        scaler: Fitted scaler
    """
    logger.info(f"\n{'='*80}")
    logger.info("PREPARING RISK CLASSIFICATION DATA")
    logger.info(f"{'='*80}\n")
    
    # Connect to database
    db_conn = psycopg2.connect(db_url)
    cursor = db_conn.cursor()
    
    # Get latest yield data for each asset
    cursor.execute("""
        WITH latest_records AS (
            SELECT asset, 
                   apy_percent,
                   COALESCE(tvl_usd, 0) as tvl,
                   COALESCE(utilization_ratio, 0) as utilization,
                   recorded_at,
                   ROW_NUMBER() OVER (PARTITION BY asset ORDER BY recorded_at DESC) as rn
            FROM protocol_yields
            WHERE apy_percent <= 200  -- Exclude extreme outliers
        )
        SELECT asset, apy_percent, tvl, utilization
        FROM latest_records
        WHERE rn = 1
    """)
    
    records = cursor.fetchall()
    db_conn.close()
    
    logger.info(f"Fetched {len(records)} unique assets")
    
    if len(records) < 10:
        raise ValueError(f"Need at least 10 assets, have {len(records)}")
    
    # Create features and labels
    X = []
    y = []
    assets = []
    
    for asset, apy, tvl, util in records:
        # Simple 4-feature model for risk classification
        features = [
            apy,
            tvl,
            util,
            apy / (tvl + 1)  # Yield/TVL ratio (risk indicator)
        ]
        
        risk_label = calculate_risk_label(apy, tvl, util)
        
        X.append(features)
        y.append(risk_label)
        assets.append(asset)
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Created {len(X)} samples with 4 features")
    logger.info(f"\nRisk distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        logger.info(f"  {label}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    logger.info(f"\nLabel encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    return X_scaled, y_encoded, y, assets, scaler, le


def train_risk_classifier(X, y, y_labels):
    """
    Train XGBoost risk classifier
    
    Args:
        X: Scaled feature matrix
        y: Encoded labels
        y_labels: Original label strings
        
    Returns:
        model: Trained XGBoost model
        metrics: Performance metrics
    """
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING XGBOOST RISK CLASSIFIER")
    logger.info(f"{'='*80}\n")
    
    # Split data (without stratify if classes are imbalanced)
    X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
        X, y, y_labels, test_size=0.25, random_state=42
    )
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # XGBoost parameters
    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y)),
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'mlogloss'
    }
    
    logger.info(f"\nModel parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    
    # Train model
    model = xgb.XGBClassifier(**params)
    
    logger.info("\nTraining model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    # XGBoost sometimes returns probabilities instead of class labels
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*80}\n")
    
    logger.info(f"✅ Accuracy: {accuracy:.3f}")
    logger.info(f"✅ F1 Score: {f1:.3f}")
    
    # Classification report
    logger.info(f"\n{'='*80}")
    logger.info("CLASSIFICATION REPORT")
    logger.info(f"{'='*80}\n")
    
    from sklearn.metrics import classification_report
    report = classification_report(labels_test, [['low', 'medium', 'high'][i] for i in y_pred])
    logger.info(report)
    
    # Feature importance
    logger.info(f"\n{'='*80}")
    logger.info("FEATURE IMPORTANCE")
    logger.info(f"{'='*80}\n")
    
    feature_names = ['APY', 'TVL', 'Utilization', 'Yield/TVL Ratio']
    importances = model.feature_importances_
    for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        logger.info(f"  {name:20s}: {importance:.3f}")
    
    # Confusion matrix
    logger.info(f"\n{'='*80}")
    logger.info("CONFUSION MATRIX")
    logger.info(f"{'='*80}\n")
    
    cm = confusion_matrix(y_test, y_pred)
    unique_labels = np.unique(y_test)
    label_names = [['Low', 'Med', 'High'][i] for i in unique_labels]
    
    logger.info(f"         Predicted")
    logger.info(f"         {' '.join([f'{l:>4s}' for l in label_names])}")
    logger.info(f"Actual")
    for i, row in enumerate(cm):
        label = label_names[i]
        row_str = '  '.join([f'{val:>4d}' for val in row])
        logger.info(f"{label:6s}   {row_str}")
    
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm
    }
    
    return model, metrics


def save_model(model, scaler, label_encoder):
    """Save trained model and preprocessing objects"""
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model.save_model('models/xgboost_risk_classifier.json')
    logger.info("✅ Saved model to: models/xgboost_risk_classifier.json")
    
    # Save scaler
    with open('models/risk_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("✅ Saved scaler to: models/risk_scaler.pkl")
    
    # Save label encoder
    with open('models/risk_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    logger.info("✅ Saved label encoder to: models/risk_label_encoder.pkl")


def main():
    """Main training pipeline"""
    load_dotenv()
    
    try:
        # Build database URL
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            db_url = f"dbname={os.getenv('DB_NAME', 'defi_yield_db')} " \
                     f"user={os.getenv('DB_USER', 'faizan')} " \
                     f"password={os.getenv('DB_PASSWORD', '')} " \
                     f"host={os.getenv('DB_HOST', 'localhost')} " \
                     f"port={os.getenv('DB_PORT', '5432')}"
        
        # Prepare data
        X, y_encoded, y_labels, assets, scaler, le = prepare_risk_data(db_url)
        
        # Train model
        model, metrics = train_risk_classifier(X, y_encoded, y_labels)
        
        # Save artifacts
        save_model(model, scaler, le)
        
        logger.info(f"\n{'='*80}")
        logger.info("✅ RISK CLASSIFIER TRAINING COMPLETE")
        logger.info(f"{'='*80}\n")
        
        logger.info("📝 Next steps:")
        logger.info("   1. Test classifier: python scripts/test_risk_classifier.py")
        logger.info("   2. Integrate with rebalancing engine")
        logger.info("   3. Build backtesting framework\n")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
