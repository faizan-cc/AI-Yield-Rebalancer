"""
Backtesting Engine for AI-Driven DeFi Yield Rebalancing System

This script simulates portfolio rebalancing strategies over historical data
and calculates performance metrics to compare ML-driven vs baseline strategies.

Phase 1 Week 9-12 Implementation
"""

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import torch
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import LSTM model class
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from src.ml.yield_predictor import YieldForecaster
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("⚠️  Could not import YieldForecaster - LSTM disabled")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Portfolio:
    """Tracks portfolio state during backtesting"""
    
    def __init__(self, initial_capital: float = 10000.0, transaction_cost: float = 0.0005, min_trade_size: float = 100.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: Dict[str, float] = {}  # asset -> amount
        self.history: List[Dict] = []
        self.transaction_cost = transaction_cost  # Default 0.05% (optimized for Uniswap V3)
        self.min_trade_size = min_trade_size  # Minimum $100 per trade to reduce small trades
        self.last_allocations: Dict[str, float] = {}  # Track last allocations for drift calculation
        
    def rebalance(self, allocations: Dict[str, float], timestamp: datetime):
        """
        Rebalance portfolio to target allocations
        
        Args:
            allocations: {asset: weight} where weights sum to 1.0
            timestamp: Current timestamp
        """
        total_value = self.get_total_value()
        
        # Clear old positions
        self.positions = {}
        
        # Calculate transaction costs and allocate
        transaction_costs = 0.0
        trades_executed = []
        allocated_amount = 0.0
        
        for asset, weight in allocations.items():
            target_amount = total_value * weight
            if target_amount >= self.min_trade_size:  # Only allocate amounts >= min_trade_size
                cost = target_amount * self.transaction_cost
                transaction_costs += cost
                self.positions[asset] = target_amount
                allocated_amount += target_amount
                trades_executed.append({
                    'asset': asset,
                    'amount': target_amount,
                    'cost': cost
                })
        
        # Remaining capital after deducting transaction costs
        # capital = total_value - allocated_amount - transaction_costs
        # But for simplicity, we'll reduce capital by transaction costs
        # and positions hold the invested amounts
        self.capital = total_value - allocated_amount - transaction_costs
        
        # Record history
        self.history.append({
            'timestamp': timestamp,
            'total_value': total_value,
            'capital': self.capital,
            'positions': self.positions.copy(),
            'transaction_costs': transaction_costs,
            'trades': trades_executed
        })
        
    def get_total_value(self) -> float:
        """Calculate total portfolio value"""
        return self.capital + sum(self.positions.values())
    
    def apply_yields(self, yields: Dict[str, float], time_delta_hours: float):
        """
        Apply yield returns to positions
        
        Args:
            yields: {asset: apy_percent}
            time_delta_hours: Hours elapsed since last update
        """
        if time_delta_hours == 0:
            return  # Skip if no time has passed
            
        for asset, amount in self.positions.items():
            if asset in yields:
                apy = yields[asset] / 100.0  # Convert to decimal
                # Calculate return for time period
                hourly_rate = (1 + apy) ** (1 / 8760) - 1  # 8760 hours/year
                period_return = (1 + hourly_rate) ** time_delta_hours - 1
                earnings = amount * period_return
                self.positions[asset] += earnings
    
    def get_performance_metrics(self) -> Dict:
        """Calculate portfolio performance metrics"""
        if len(self.history) < 2:
            return {}
        
        # Extract value series
        values = [h['total_value'] for h in self.history]
        timestamps = [h['timestamp'] for h in self.history]
        
        # Total return
        total_return = (values[-1] - self.initial_capital) / self.initial_capital
        
        # Calculate returns
        returns = np.diff(values) / values[:-1]
        
        # Annualized return (only if test period >= 30 days)
        days_elapsed = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
        if days_elapsed >= 30:  # Only annualize if >= 30 days
            try:
                annualized_return = (1 + total_return) ** (365 / days_elapsed) - 1
            except OverflowError:
                annualized_return = total_return * (365 / days_elapsed)
        else:
            # For short backtests, report period return (not annualized)
            annualized_return = total_return
        
        # Volatility (annualized if >= 30 days, else period volatility)
        intervals_per_day = 288  # 5-minute intervals in a day
        if len(returns) > 1:
            period_volatility = np.std(returns)
            if days_elapsed >= 30:
                # Annualize based on frequency (5-min intervals = 288 per day)
                volatility = period_volatility * np.sqrt(365 * intervals_per_day)
            else:
                # Report period volatility
                volatility = period_volatility * np.sqrt(intervals_per_day) if days_elapsed > 0 else period_volatility
        else:
            volatility = 0.0
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02 / 365 if days_elapsed < 30 else 0.02  # Daily rate for short periods
        if volatility > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        cumulative = np.array(values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0.0
        
        # Total transaction costs
        total_costs = sum(h['transaction_costs'] for h in self.history)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_value': values[-1],
            'total_transaction_costs': total_costs,
            'num_rebalances': len(self.history),
            'days_elapsed': days_elapsed
        }


class MLStrategy:
    """ML-driven rebalancing strategy using LSTM + XGBoost"""
    
    def __init__(self):
        self.lstm_model = None
        self.xgboost_model = None
        self.feature_scaler = None
        self.risk_scaler = None
        self.label_encoder = None
        self.sequence_length = 7
        self.load_models()
        
    def load_models(self):
        """Load trained ML models"""
        logger.info("Loading ML models...")
        
        # Load LSTM for yield prediction (PyTorch Lightning checkpoint)
        if LSTM_AVAILABLE:
            try:
                import pytorch_lightning as pl
                self.lstm_model = YieldPredictorLSTM.load_from_checkpoint(
                    'models/lstm_predictor_final.ckpt',
                    map_location='cpu'
                )
                self.lstm_model.eval()
                logger.info("✅ Loaded LSTM yield predictor")
            except Exception as e:
                logger.warning(f"⚠️  Could not load LSTM model: {e}")
                self.lstm_model = None
        else:
            logger.warning("⚠️  LSTM module not available - skipping LSTM load")
            self.lstm_model = None
        
        # Load XGBoost
        try:
            self.xgboost_model = XGBClassifier()
            self.xgboost_model.load_model('models/xgboost_risk_classifier.json')
            logger.info("✅ Loaded XGBoost model")
        except Exception as e:
            logger.warning(f"⚠️  Could not load XGBoost model: {e}")
        
        # Load feature scaler for LSTM
        try:
            with open('models/feature_scaler.pkl', 'rb') as f:
                self.feature_scaler = pickle.load(f)
            logger.info("✅ Loaded feature scaler")
        except Exception as e:
            logger.warning(f"⚠️  Could not load feature scaler: {e}")
            self.feature_scaler = None
        
        # Load risk scaler and label encoder
        try:
            with open('models/risk_scaler.pkl', 'rb') as f:
                self.risk_scaler = pickle.load(f)
            with open('models/risk_label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            logger.info("✅ Loaded risk preprocessing")
        except Exception as e:
            logger.warning(f"⚠️  Could not load risk preprocessing: {e}")
    
    def predict_yields(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Predict next period yields using LSTM
        
        Args:
            historical_data: {asset: DataFrame with columns [apy, tvl, utilization, risk]}
            
        Returns:
            {asset: predicted_apy}
        """
        if self.lstm_model is None or self.feature_scaler is None:
            return {}
        
        predictions = {}
        
        for asset, df in historical_data.items():
            if len(df) < self.sequence_length:
                continue
            
            # Prepare sequence
            features = df[['apy_percent', 'tvl', 'utilization_rate', 'risk_score']].values
            features_scaled = self.feature_scaler.transform(features[-self.sequence_length:])
            
            # Reshape for LSTM: (1, sequence_length, num_features)
            sequence = torch.FloatTensor(features_scaled).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                pred = self.lstm_model(sequence)
                # Handle both single tensor and tuple outputs
                if isinstance(pred, tuple):
                    pred = pred[0]
                predicted_apy = pred.item()
            
            predictions[asset] = predicted_apy
        
        return predictions
    
    def predict_risks(self, current_data: Dict[str, Dict]) -> Dict[str, str]:
        """
        Classify risk levels using XGBoost
        
        Args:
            current_data: {asset: {apy, tvl, utilization}}
            
        Returns:
            {asset: risk_level}
        """
        if self.xgboost_model is None or self.risk_scaler is None:
            return {}
        
        risks = {}
        
        for asset, data in current_data.items():
            # Prepare features
            features = np.array([[
                data['apy'],
                data['tvl'],
                data['utilization'],
                data['apy'] / (data['tvl'] + 1)
            ]])
            
            # Scale and predict
            features_scaled = self.risk_scaler.transform(features)
            risk_pred = self.xgboost_model.predict(features_scaled)
            
            # Handle both class labels and probability arrays
            if len(risk_pred.shape) > 1:
                risk_encoded = np.argmax(risk_pred, axis=1)[0]
            else:
                risk_encoded = risk_pred[0]
            
            risk_label = self.label_encoder.inverse_transform([int(risk_encoded)])[0]
            
            risks[asset] = risk_label
        
        return risks
    
    def calculate_allocations(self, 
                            yield_predictions: Dict[str, float],
                            risk_classifications: Dict[str, str],
                            current_data: Dict[str, Dict],
                            min_apy_threshold: float = 50.0) -> Dict[str, float]:
        """
        Calculate optimal portfolio allocations
        
        Strategy:
        1. Filter out high-risk assets
        2. Filter assets below minimum APY threshold (50%)
        3. Score assets by current APY / risk
        4. Allocate proportionally to scores
        
        Returns:
            {asset: weight} where weights sum to 1.0
        """
        # Risk weights
        risk_weights = {
            'high': 0.0,    # Exclude high-risk
            'medium': 1.0,
            'low': 1.5      # Prefer low-risk
        }
        
        # Calculate scores using current APY (not predictions since LSTM unavailable)
        scores = {}
        for asset in current_data.keys():
            if asset not in risk_classifications:
                continue
            
            risk = risk_classifications[asset]
            risk_weight = risk_weights.get(risk, 0.0)
            
            if risk_weight == 0.0:
                continue
            
            current_apy = current_data[asset]['apy']
            
            # Filter out low-yield assets (< 50% APY)
            if current_apy < min_apy_threshold:
                continue
            
            # Score = current_apy * risk_weight
            score = max(0, current_apy) * risk_weight
            scores[asset] = score
        
        # Normalize to weights
        total_score = sum(scores.values())
        if total_score == 0:
            # Fallback: equal weight
            allocations = {asset: 1.0 / len(scores) for asset in scores.keys()}
        else:
            allocations = {asset: score / total_score 
                          for asset, score in scores.items()}
        
        return allocations


class BaselineStrategy:
    """Baseline rebalancing strategies for comparison"""
    
    @staticmethod
    def equal_weight(assets: List[str]) -> Dict[str, float]:
        """Equal allocation across all assets"""
        weight = 1.0 / len(assets) if assets else 0.0
        return {asset: weight for asset in assets}
    
    @staticmethod
    def best_historical(current_data: Dict[str, Dict]) -> Dict[str, float]:
        """Allocate to top 3 assets by current APY"""
        # Sort by APY
        sorted_assets = sorted(current_data.items(), 
                              key=lambda x: x[1]['apy'], 
                              reverse=True)
        
        # Top 3 get equal weight
        top_assets = [asset for asset, _ in sorted_assets[:3]]
        if not top_assets:
            return {}
        
        weight = 1.0 / len(top_assets)
        allocations = {asset: weight for asset in top_assets}
        
        return allocations
    
    @staticmethod
    def highest_tvl(current_data: Dict[str, Dict]) -> Dict[str, float]:
        """Allocate to top 3 assets by TVL (lowest risk proxy)"""
        sorted_assets = sorted(current_data.items(),
                              key=lambda x: x[1]['tvl'],
                              reverse=True)
        
        top_assets = [asset for asset, _ in sorted_assets[:3]]
        if not top_assets:
            return {}
        
        weight = 1.0 / len(top_assets)
        allocations = {asset: weight for asset in top_assets}
        
        return allocations


def load_historical_data(conn) -> pd.DataFrame:
    """Load all historical yield data from database"""
    query = """
        SELECT 
            protocol_id,
            asset,
            apy_percent,
            total_liquidity_usd as tvl,
            utilization_ratio as utilization_rate,
            50 as risk_score,
            recorded_at as timestamp
        FROM protocol_yields
        ORDER BY recorded_at ASC
    """
    
    df = pd.read_sql(query, conn)
    
    # Round timestamps to nearest 5 minutes for grouping
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = df['timestamp'].dt.floor('5T')  # 5 minute intervals
    
    # Group by timestamp and asset, take mean of metrics
    df = df.groupby(['timestamp', 'asset', 'protocol_id']).agg({
        'apy_percent': 'mean',
        'tvl': 'mean',
        'utilization_rate': 'mean',
        'risk_score': 'mean'
    }).reset_index()
    
    logger.info(f"Loaded {len(df)} historical data points across {df['timestamp'].nunique()} time periods")
    
    return df


def run_backtest(strategy_name: str, 
                 strategy_func,
                 historical_data: pd.DataFrame,
                 rebalance_frequency_hours: float = 8.0,
                 min_rebalance_threshold: float = 0.05) -> Portfolio:
    """
    Run backtest for a given strategy
    
    Args:
        strategy_name: Name of strategy
        strategy_func: Function that returns allocations
        historical_data: DataFrame with all historical data
        rebalance_frequency_hours: How often to rebalance (default 8h for profitability)
        min_rebalance_threshold: Minimum portfolio drift to trigger rebalance (default 5%)
        
    Returns:
        Portfolio with complete history
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"BACKTESTING: {strategy_name}")
    logger.info(f"{'='*80}")
    
    portfolio = Portfolio(initial_capital=10000.0)
    
    # Get unique timestamps
    timestamps = sorted(historical_data['timestamp'].unique())
    
    if len(timestamps) < 2:
        logger.warning("⚠️  Insufficient data for backtesting")
        return portfolio
    
    last_rebalance_time = timestamps[0]
    prev_timestamp = timestamps[0]
    
    for i, current_time in enumerate(timestamps):
        # Get current data snapshot
        current_snapshot = historical_data[
            historical_data['timestamp'] == current_time
        ]
        
        # Prepare current data dict
        current_data = {}
        for _, row in current_snapshot.iterrows():
            asset = row['asset']
            current_data[asset] = {
                'apy': row['apy_percent'],
                'tvl': row['tvl'],
                'utilization': row['utilization_rate'],
                'risk': row['risk_score']
            }
        
        # Apply yields for time period (if not first iteration)
        if i > 0:
            time_delta = (current_time - prev_timestamp).total_seconds() / 3600
            
            # Get yields for current positions
            yields = {asset: current_data[asset]['apy'] 
                     for asset in portfolio.positions.keys()
                     if asset in current_data}
            
            if time_delta > 0 and yields:
                # Debug logging for first few yield applications
                if i <= 3 or i % 30 == 0:
                    logger.debug(f"Step {i}: Applying yields - time_delta={time_delta:.6f}h, positions={list(yields.keys())}, portfolio_value=${portfolio.get_total_value():,.2f}")
            
            portfolio.apply_yields(yields, time_delta)
        
        # Check if it's time to rebalance
        time_since_rebalance = (current_time - last_rebalance_time).total_seconds() / 3600
        
        # Calculate allocation drift if we have positions
        should_rebalance = (i == 0 or time_since_rebalance >= rebalance_frequency_hours)
        
        if not should_rebalance and portfolio.positions and portfolio.last_allocations and min_rebalance_threshold > 0:
            total_val = portfolio.get_total_value()
            max_drift = 0.0
            
            # Calculate drift from target allocations
            for asset in portfolio.last_allocations.keys():
                target_weight = portfolio.last_allocations[asset]
                current_weight = portfolio.positions.get(asset, 0) / total_val if total_val > 0 else 0
                drift = abs(current_weight - target_weight)
                max_drift = max(max_drift, drift)
            
            # Trigger rebalance if drift exceeds threshold
            if max_drift >= min_rebalance_threshold:
                should_rebalance = True
                logger.info(f"Step {i}: Drift threshold triggered ({max_drift:.2%} > {min_rebalance_threshold:.2%})")
        
        if should_rebalance:
            # Get allocations from strategy
            if strategy_name == "ML-Driven":
                # For ML strategy, need historical sequences
                historical_sequences = {}
                for asset in current_data.keys():
                    asset_history = historical_data[
                        (historical_data['asset'] == asset) &
                        (historical_data['timestamp'] <= current_time)
                    ].tail(7)  # Last 7 data points
                    
                    if len(asset_history) >= 7:
                        historical_sequences[asset] = asset_history
                
                # Get predictions
                yield_preds = strategy_func.predict_yields(historical_sequences)
                risk_preds = strategy_func.predict_risks(current_data)
                
                logger.info(f"Step {i}: Risk classifications: {len(risk_preds)} assets")
                
                allocations = strategy_func.calculate_allocations(
                    yield_preds, risk_preds, current_data
                )
            else:
                # Baseline strategies
                allocations = strategy_func(current_data)
            
            logger.info(f"Step {i}: Allocations for {len(allocations)} assets: {allocations}")
            
            # Rebalance portfolio
            if allocations:
                portfolio.rebalance(allocations, current_time)
                last_rebalance_time = current_time
        else:
            # Record current portfolio value even if not rebalancing
            portfolio.history.append({
                'timestamp': current_time,
                'total_value': portfolio.get_total_value(),
                'capital': portfolio.capital,
                'positions': portfolio.positions.copy(),
                'transaction_costs': 0.0,
                'trades': []
            })
        
        # Update prev_timestamp for next iteration
        prev_timestamp = current_time
    
    # Calculate metrics
    metrics = portfolio.get_performance_metrics()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"RESULTS: {strategy_name}")
    logger.info(f"{'='*80}")
    logger.info(f"Initial Capital    : ${portfolio.initial_capital:,.2f}")
    logger.info(f"Final Value        : ${metrics.get('final_value', 0):,.2f}")
    logger.info(f"Total Return       : {metrics.get('total_return', 0)*100:.2f}%")
    
    # Label return appropriately based on period length
    days = metrics.get('days_elapsed', 0)
    if days >= 30:
        logger.info(f"Annualized Return  : {metrics.get('annualized_return', 0)*100:.2f}%")
        logger.info(f"Volatility (Ann.)  : {metrics.get('volatility', 0)*100:.2f}%")
    else:
        logger.info(f"Period Return ({days:.1f}d): {metrics.get('annualized_return', 0)*100:.2f}%")
        logger.info(f"Period Volatility  : {metrics.get('volatility', 0)*100:.2f}%")
    
    logger.info(f"Sharpe Ratio       : {metrics.get('sharpe_ratio', 0):.6f}")
    logger.info(f"Max Drawdown       : {metrics.get('max_drawdown', 0)*100:.2f}%")
    logger.info(f"Win Rate           : {metrics.get('win_rate', 0)*100:.2f}%")
    logger.info(f"Transaction Costs  : ${metrics.get('total_transaction_costs', 0):,.2f}")
    logger.info(f"Num Rebalances     : {metrics.get('num_rebalances', 0)}")
    logger.info(f"Days Elapsed       : {metrics.get('days_elapsed', 0):.1f}")
    
    return portfolio


def main():
    """Main backtesting execution"""
    logger.info("="*80)
    logger.info("DEFI YIELD REBALANCING - BACKTESTING ENGINE")
    logger.info("="*80)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Connect to database
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME', 'defi_yield_db'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'postgres'),
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432')
    )
    
    try:
        # Load historical data
        historical_data = load_historical_data(conn)
        
        if len(historical_data) < 10:
            logger.error("❌ Insufficient data for backtesting. Need at least 10 records.")
            return
        
        # Initialize strategies
        ml_strategy = MLStrategy()
        
        # Run backtests
        results = {}
        
        # 1. ML-Driven Strategy (Optimized for Profitability)
        results['ML-Driven'] = run_backtest(
            "ML-Driven",
            ml_strategy,
            historical_data,
            rebalance_frequency_hours=4.0  # Reduce from 24h to 4h for better responsiveness
        )
        
        # 2. Equal Weight Baseline
        results['Equal-Weight'] = run_backtest(
            "Equal-Weight",
            lambda data: BaselineStrategy.equal_weight(list(data.keys())),
            historical_data,
            rebalance_frequency_hours=4.0
        )
        
        # 3. Best Historical APY
        results['Best-Historical-APY'] = run_backtest(
            "Best-Historical-APY",
            BaselineStrategy.best_historical,
            historical_data,
            rebalance_frequency_hours=4.0
        )
        
        # 4. Highest TVL (Safety)
        results['Highest-TVL'] = run_backtest(
            "Highest-TVL",
            BaselineStrategy.highest_tvl,
            historical_data,
            rebalance_frequency_hours=4.0
        )
        
        # Comparison table
        logger.info(f"\n{'='*80}")
        logger.info("STRATEGY COMPARISON")
        logger.info(f"{'='*80}\n")
        
        comparison_df = pd.DataFrame({
            name: portfolio.get_performance_metrics()
            for name, portfolio in results.items()
        }).T
        
        logger.info(comparison_df.to_string())
        
        # Save results
        comparison_df.to_csv('backtest_results.csv')
        logger.info(f"\n✅ Results saved to: backtest_results.csv")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
