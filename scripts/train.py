"""
Train LSTM Model on DeFi Yield Data
Uses new database schema: assets + yield_metrics tables
"""

import os
import sys
import logging
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data_from_db():
    """Load data from new database schema"""
    
    logger.info("="*60)
    logger.info("Loading data from database...")
    logger.info("="*60)
    
    db_url = os.getenv('DATABASE_URL')
    conn = psycopg2.connect(db_url)
    
    query = """
        SELECT 
            ym.time,
            ym.asset_id,
            a.protocol,
            a.symbol,
            ym.apy_percent,
            ym.tvl_usd,
            COALESCE(ym.volume_24h_usd, 0) as volume_24h_usd,
            COALESCE(ym.volatility_24h, 0) as volatility_24h
        FROM yield_metrics ym
        JOIN assets a ON ym.asset_id = a.id
        ORDER BY ym.asset_id, ym.time
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    logger.info(f"✓ Loaded {len(df):,} records")
    logger.info(f"  Date range: {df['time'].min()} to {df['time'].max()}")
    logger.info(f"  Assets: {df['asset_id'].nunique()}")
    logger.info(f"  Protocols: {', '.join(df['protocol'].unique())}")
    
    return df


def create_features(df, lookback=30):
    """Create time-series features"""
    
    logger.info("\n" + "="*60)
    logger.info("Creating features...")
    logger.info("="*60)
    
    df = df.copy()
    df = df.sort_values(['asset_id', 'time'])
    
    # Temporal features
    df['hour'] = pd.to_datetime(df['time']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['time']).dt.dayofweek
    
    # Protocol encoding
    protocol_dummies = pd.get_dummies(df['protocol'], prefix='proto')
    df = pd.concat([df, protocol_dummies], axis=1)
    
    # Per-asset rolling features
    feature_dfs = []
    
    for asset_id in df['asset_id'].unique():
        asset_df = df[df['asset_id'] == asset_id].copy().sort_values('time')
        
        # Lag features
        asset_df['apy_lag_1d'] = asset_df['apy_percent'].shift(1)
        asset_df['apy_lag_7d'] = asset_df['apy_percent'].shift(7)
        
        # Rolling means
        asset_df['apy_ma_7d'] = asset_df['apy_percent'].rolling(7, min_periods=1).mean()
        asset_df['apy_ma_30d'] = asset_df['apy_percent'].rolling(30, min_periods=1).mean()
        asset_df['tvl_ma_7d'] = asset_df['tvl_usd'].rolling(7, min_periods=1).mean()
        
        # Rolling std
        asset_df['apy_std_7d'] = asset_df['apy_percent'].rolling(7, min_periods=1).std()
        asset_df['apy_std_30d'] = asset_df['apy_percent'].rolling(30, min_periods=1).std()
        
        # Trends
        asset_df['apy_trend_7d'] = asset_df['apy_percent'].pct_change(7) * 100
        asset_df['tvl_trend_7d'] = asset_df['tvl_usd'].pct_change(7) * 100
        
        # Volume ratio
        asset_df['volume_tvl_ratio'] = asset_df['volume_24h_usd'] / (asset_df['tvl_usd'] + 1e-10)
        
        # Momentum
        asset_df['apy_momentum'] = asset_df['apy_ma_7d'] - asset_df['apy_ma_30d']
        
        feature_dfs.append(asset_df)
    
    df_features = pd.concat(feature_dfs, ignore_index=True)
    df_features = df_features.fillna(method='bfill').fillna(0)
    
    logger.info(f"✓ Created {len(df_features.columns)} total columns")
    
    return df_features


def prepare_sequences(df, sequence_length=30, prediction_horizon=7):
    """Prepare sequences for LSTM"""
    
    logger.info("\n" + "="*60)
    logger.info("Preparing sequences...")
    logger.info("="*60)
    
    # Select feature columns
    feature_cols = [
        'apy_percent', 'tvl_usd', 'volume_24h_usd', 'volatility_24h',
        'apy_ma_7d', 'apy_ma_30d', 'apy_std_7d',
        'tvl_ma_7d', 'tvl_trend_7d', 'volume_tvl_ratio',
        'apy_momentum', 'hour', 'day_of_week'
    ]
    
    # Add protocol columns
    protocol_cols = [c for c in df.columns if c.startswith('proto_')]
    feature_cols.extend(protocol_cols)
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    logger.info(f"Using {len(feature_cols)} features")
    
    X_list, y_list = [], []
    
    for asset_id in df['asset_id'].unique():
        asset_df = df[df['asset_id'] == asset_id].sort_values('time').reset_index(drop=True)
        
        if len(asset_df) < sequence_length + prediction_horizon:
            continue
        
        features = asset_df[feature_cols].values
        targets = asset_df['apy_percent'].values
        
        for i in range(len(asset_df) - sequence_length - prediction_horizon + 1):
            X_list.append(features[i:i + sequence_length])
            y_list.append(targets[i + sequence_length + prediction_horizon - 1])
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    logger.info(f"✓ Created {len(X):,} sequences")
    logger.info(f"  X shape: {X.shape}")
    logger.info(f"  y shape: {y.shape}")
    
    return X, y, feature_cols


def train_model(X, y):
    """Train LSTM model"""
    
    logger.info("\n" + "="*60)
    logger.info("Training LSTM model...")
    logger.info("="*60)
    
    # Split data (80/10/10)
    n = len(X)
    train_idx = int(n * 0.8)
    val_idx = int(n * 0.9)
    
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]
    
    logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Normalize
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_flat)
    
    X_train = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    with open('models/feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("✓ Saved scaler")
    
    # Create datasets
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Build model
    from src.ml.yield_predictor import YieldForecaster
    
    input_size = X_train.shape[2]
    model = YieldForecaster(input_size=input_size, hidden_size_1=128, hidden_size_2=64)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    logger.info(f"Device: {device}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    for epoch in range(100):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred.squeeze(), y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/lstm_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('models/lstm_best.pth'))
    
    # Test
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            predictions.extend(y_pred.cpu().numpy())
            actuals.extend(y_batch.numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals)
    
    # Metrics
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mae = np.mean(np.abs(predictions - actuals))
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-10))) * 100
    
    logger.info("\n" + "="*60)
    logger.info("Test Results")
    logger.info("="*60)
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE:  {mae:.4f}")
    logger.info(f"MAPE: {mape:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), 'models/lstm_yield_predictor.pth')
    logger.info(f"\n✓ Model saved to models/lstm_yield_predictor.pth")
    
    return model


def main():
    load_dotenv()
    
    try:
        # Load and prepare data
        df = load_data_from_db()
        df_features = create_features(df)
        X, y, feature_cols = prepare_sequences(df_features)
        
        # Train model
        model = train_model(X, y)
        
        logger.info("\n" + "="*60)
        logger.info("✅ Training Complete!")
        logger.info("="*60)
        logger.info("Files created:")
        logger.info("  • models/lstm_yield_predictor.pth")
        logger.info("  • models/lstm_best.pth")
        logger.info("  • models/feature_scaler.pkl")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
