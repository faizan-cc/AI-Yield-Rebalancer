"""
Training Script for LSTM Yield Predictor
Trains model on historical protocol data
"""

import os
import sys
import logging
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.feature_engineering import FeatureEngineer
from src.ml.lstm_predictor import YieldPredictorLSTM, create_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_training_data(db_url: str, lookback_days: int = 90):
    """
    Prepare training data from database
    
    Args:
        db_url: Database connection string
        lookback_days: Number of days of historical data to use
        
    Returns:
        features, targets, protocols, scaler
    """
    logger.info(f"Preparing training data ({lookback_days} days lookback)")
    
    # Connect to database
    db_conn = psycopg2.connect(db_url)
    engineer = FeatureEngineer(db_conn)
    
    # Get protocols
    cursor = db_conn.cursor()
    cursor.execute("SELECT id, name FROM protocols ORDER BY id")
    protocols = cursor.fetchall()
    cursor.close()
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Create training dataset
    feature_matrix, feature_objects = engineer.create_training_dataset(
        start_date=start_date,
        end_date=end_date,
        protocols=protocols
    )
    
    db_conn.close()
    
    if len(feature_matrix) == 0:
        raise ValueError("No training data created. Need more historical data.")
    
    # Extract targets (current APY) - we'll predict 7 days ahead
    targets = np.array([f.current_apy for f in feature_objects])
    
    logger.info(f"Created {len(feature_matrix)} samples")
    logger.info(f"Feature shape: {feature_matrix.shape}")
    logger.info(f"Target shape: {targets.shape}")
    logger.info(f"Target range: {targets.min():.3f}% to {targets.max():.3f}%")
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(feature_matrix)
    
    return features_normalized, targets, feature_objects, scaler


def train_model(
    features: np.ndarray,
    targets: np.ndarray,
    epochs: int = 50,
    batch_size: int = 16,
    sequence_length: int = 14,
):
    """
    Train LSTM yield predictor
    
    Args:
        features: Normalized feature matrix
        targets: Target APYs
        epochs: Number of training epochs
        batch_size: Batch size
        sequence_length: LSTM sequence length
    """
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING LSTM YIELD PREDICTOR")
    logger.info(f"{'='*80}\n")
    
    # Split data
    train_features, val_features, train_targets, val_targets = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training samples: {len(train_features)}")
    logger.info(f"Validation samples: {len(val_features)}")
    
    # Check if we have enough data for sequences
    if len(train_features) < sequence_length + 1:
        logger.warning(f"Not enough data for sequence length {sequence_length}")
        logger.warning(f"Need at least {sequence_length + 1} samples, have {len(train_features)}")
        logger.warning("Reducing sequence length to 3")
        sequence_length = 3
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_features, train_targets,
        val_features, val_targets,
        sequence_length=sequence_length,
        batch_size=batch_size
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Initialize model
    model = YieldPredictorLSTM(
        input_size=32,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        learning_rate=0.001
    )
    
    logger.info(f"\nModel architecture:")
    logger.info(f"  Input size: 32 features")
    logger.info(f"  Hidden size: 128")
    logger.info(f"  LSTM layers: 2")
    logger.info(f"  Attention: Yes")
    logger.info(f"  Dropout: 0.3")
    logger.info(f"  Learning rate: 0.001")
    
    # Callbacks
    # Determine monitoring metric based on validation set availability
    has_val_data = val_loader is not None and len(val_loader) > 0
    monitor_metric = 'val_loss' if has_val_data else 'train_loss'
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/checkpoints',
        filename='lstm-{epoch:02d}-{' + monitor_metric + ':.4f}',
        save_top_k=3,
        monitor=monitor_metric,
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=10,
        mode='min'
    )
    
    # Logger
    tb_logger = TensorBoardLogger('models/logs', name='lstm_predictor')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=tb_logger,
        accelerator='auto',
        devices=1,
        log_every_n_steps=10,
        deterministic=True
    )
    
    # Train
    logger.info(f"\n{'='*80}")
    logger.info("Starting training...")
    logger.info(f"{'='*80}\n")
    
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    os.makedirs('models', exist_ok=True)
    trainer.save_checkpoint('models/lstm_predictor_final.ckpt')
    
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"✅ Model saved to: models/lstm_predictor_final.ckpt")
    logger.info(f"✅ Best checkpoint: {checkpoint_callback.best_model_path}")
    logger.info(f"✅ TensorBoard logs: models/logs/")
    
    # Test model
    logger.info(f"\n{'='*80}")
    logger.info("Testing model on validation set...")
    logger.info(f"{'='*80}\n")
    
    test_results = trainer.test(model, val_loader)
    
    if test_results:
        result = test_results[0]
        logger.info(f"\nTest Results:")
        logger.info(f"  Loss: {result.get('test_loss', 0):.4f}")
        logger.info(f"  MAPE: {result.get('test_mape', 0):.2f}%")
        logger.info(f"  MAE: {result.get('test_mae', 0):.4f}%")
        logger.info(f"  RMSE: {result.get('test_rmse', 0):.4f}%")
    
    return model, trainer


def main():
    """Main training script"""
    load_dotenv()
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not configured")
        return
    
    try:
        # Prepare data
        features, targets, feature_objects, scaler = prepare_training_data(
            db_url, lookback_days=90
        )
        
        # Save scaler for inference
        import joblib
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/feature_scaler.pkl')
        logger.info("✅ Saved feature scaler to models/feature_scaler.pkl")
        
        # Train model
        model, trainer = train_model(
            features, targets,
            epochs=50,
            batch_size=16,
            sequence_length=14
        )
        
        logger.info("\n📝 Next steps:")
        logger.info("   1. View training progress: tensorboard --logdir models/logs/")
        logger.info("   2. Test predictions: python scripts/test_lstm.py")
        logger.info("   3. Deploy for inference: python scripts/inference.py")
        
    except ValueError as e:
        logger.error(f"❌ {e}")
        logger.info("\n💡 Solution: Collect more data first")
        logger.info("   Run: python scripts/collect_data.py")
        logger.info("   Schedule collection every 15 minutes for better training")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
