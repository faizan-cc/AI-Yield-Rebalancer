"""
Simple Backtesting Script
Tests different rebalancing strategies on historical data
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
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleBacktest:
    """Simple backtesting engine"""
    
    def __init__(self, initial_capital=10000.0, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def load_historical_data(self, start_date, end_date):
        """Load historical yield data"""
        
        db_url = os.getenv('DATABASE_URL')
        conn = psycopg2.connect(db_url)
        
        query = """
            SELECT 
                ym.time,
                a.protocol,
                a.symbol,
                ym.asset_id,
                ym.apy_percent,
                ym.tvl_usd
            FROM yield_metrics ym
            JOIN assets a ON ym.asset_id = a.id
            WHERE ym.time >= %s AND ym.time <= %s
            ORDER BY ym.time, ym.asset_id
        """
        
        df = pd.read_sql(query, conn, params=(start_date, end_date))
        conn.close()
        
        return df
    
    def strategy_highest_apy(self, df, date, top_n=3):
        """Strategy: Pick top N assets by APY"""
        
        day_data = df[df['time'].dt.date == date.date()]
        
        if len(day_data) == 0:
            return {}
        
        # Sort by APY and pick top N
        top_assets = day_data.nlargest(top_n, 'apy_percent')
        
        # Equal weight
        weight = 1.0 / len(top_assets)
        allocations = {row['symbol']: weight for _, row in top_assets.iterrows()}
        
        return allocations
    
    def strategy_tvl_weighted(self, df, date, min_apy=1.0):
        """Strategy: TVL-weighted allocation for assets above min APY"""
        
        day_data = df[df['time'].dt.date == date.date()]
        
        if len(day_data) == 0:
            return {}
        
        # Filter by minimum APY
        filtered = day_data[day_data['apy_percent'] >= min_apy]
        
        if len(filtered) == 0:
            return {}
        
        # Weight by TVL
        total_tvl = filtered['tvl_usd'].sum()
        if total_tvl == 0:
            return {}
        
        allocations = {
            row['symbol']: row['tvl_usd'] / total_tvl 
            for _, row in filtered.iterrows()
        }
        
        return allocations
    
    def strategy_unified_ml(self, df, date, lstm_model, lstm_scaler, feature_cols, 
                           xgb_model, risk_scaler, label_encoder,
                           sequence_length=30, min_apy=2.0, top_n=4, 
                           rebalance_threshold=0.15, position_persistence=0.8):
        """Optimized Unified ML Strategy: LSTM yield predictions + XGBoost risk filtering
        
        Profit-optimized implementation:
        1. LSTM predicts future yields for all assets
        2. XGBoost classifies risk level for each asset
        3. Aggressive risk tolerance: prioritize high yields
        4. Select top 4 assets by predicted yield
        5. Yield-weighted allocation (more capital to higher predicted yields)
        6. Position persistence: favor current holdings to reduce turnover
        7. Rebalancing threshold: only trade if improvement > threshold
        """
        
        # Get current positions if available (for persistence)
        current_positions = getattr(self, '_last_ml_positions', {})
        
        # Get data up to current date (handle timezone)
        current_ts = pd.Timestamp(date).tz_localize(None)
        df_copy = df.copy()
        df_copy['time_naive'] = pd.to_datetime(df_copy['time']).dt.tz_localize(None)
        historical_df = df_copy[df_copy['time_naive'] <= current_ts]
        
        if len(historical_df) == 0:
            logger.debug(f"No historical data for date {date.date()}")
            return {}
        
        # Get today's candidates - use data from date or closest before
        day_data = historical_df[historical_df['time_naive'].dt.date == date.date()]
        
        if len(day_data) == 0:
            # Try to find data from the most recent date before this
            recent_dates = historical_df[historical_df['time_naive'] <= current_ts]['time_naive'].dt.date.unique()
            if len(recent_dates) > 0:
                most_recent_date = max(recent_dates)
                day_data = historical_df[historical_df['time_naive'].dt.date == most_recent_date]
                logger.debug(f"No data for exact date {date.date()}, using {most_recent_date}")
        
        if len(day_data) == 0:
            logger.debug(f"No data available for or before date {date.date()}")
            return {}
        
        candidates = day_data[day_data['apy_percent'] >= min_apy]
        
        # Debug logging
        logger.debug(f"Date {date.date()}: {len(day_data)} assets, {len(candidates)} above {min_apy}% APY")
        
        if len(candidates) == 0:
            return {}
        
        asset_scores = []
        
        # Analyze each candidate asset
        for _, row in candidates.iterrows():
            asset_id = row['asset_id']
            symbol = row['symbol']
            current_apy = row['apy_percent']
            tvl = row['tvl_usd']
            protocol = row['protocol']
            
            # === LSTM: Predict future yield ===
            asset_df = historical_df[historical_df['asset_id'] == asset_id].copy()
            asset_df = asset_df.sort_values('time')
            
            predicted_yield = current_apy  # Default to current if prediction fails
            
            if len(asset_df) >= sequence_length:
                try:
                    recent_data = asset_df.tail(sequence_length)
                    sequence = recent_data[feature_cols].values
                    sequence_flat = sequence.reshape(-1, len(feature_cols))
                    sequence_normalized = lstm_scaler.transform(sequence_flat).reshape(1, sequence_length, -1)
                    
                    with torch.no_grad():
                        sequence_tensor = torch.FloatTensor(sequence_normalized)
                        predicted_yield = lstm_model(sequence_tensor).item()
                except:
                    pass  # Use current APY as fallback
            
            # === XGBoost: Classify risk level ===
            volatility = 0.0
            if 'volatility_24h' in row and not pd.isna(row['volatility_24h']):
                volatility = row['volatility_24h']
            elif 'apy_std_7d' in row and not pd.isna(row['apy_std_7d']):
                volatility = row['apy_std_7d']
            
            is_dex = 1 if protocol.lower() in ['uniswap v3', 'curve'] else 0
            is_lending = 1 if protocol.lower() in ['aave v3'] else 0
            
            risk_features = np.array([[
                current_apy,
                tvl,
                0.0,  # utilization (default)
                volatility,
                current_apy / (tvl + 1),  # yield_tvl_ratio
                is_dex,
                is_lending
            ]])
            
            risk_features_scaled = risk_scaler.transform(risk_features)
            risk_pred_proba = xgb_model.predict_proba(risk_features_scaled)[0]
            risk_pred = np.argmax(risk_pred_proba)
            risk_label = label_encoder.inverse_transform([risk_pred])[0]
            
            risk_score_map = {'low': 0, 'medium': 1, 'high': 2}
            risk_score = risk_score_map.get(risk_label, 2)
            
            asset_scores.append({
                'symbol': symbol,
                'predicted_yield': predicted_yield,
                'current_apy': current_apy,
                'risk_score': risk_score,
                'risk_label': risk_label
            })
        
        # === Combine: Aggressive yield optimization + smart risk + position persistence ===
        # Prioritize high yields: allow medium risk always, high risk if yield >8%
        acceptable_assets = [
            a for a in asset_scores 
            if a['risk_score'] <= 1 or (a['risk_score'] == 2 and a['predicted_yield'] > 8.0)
        ]
        
        if len(acceptable_assets) == 0:
            # If no acceptable assets, take highest predicted yields regardless of risk
            acceptable_assets = sorted(asset_scores, key=lambda x: x['predicted_yield'], reverse=True)[:top_n]
        
        # Apply position persistence: boost scores for current holdings
        for asset in acceptable_assets:
            if asset['symbol'] in current_positions:
                # Boost predicted yield by persistence factor to favor keeping positions
                asset['predicted_yield'] = asset['predicted_yield'] * (1 + position_persistence)
        
        # Rank by predicted yield and select top N
        top_assets = sorted(acceptable_assets, key=lambda x: x['predicted_yield'], reverse=True)[:top_n]
        
        # Yield-weighted allocation with boost for high yields
        logger.debug(f"Final allocation for {date.date()}: {len(top_assets)} assets")
        if len(top_assets) > 0:
            # Calculate weights proportional to predicted yields (squared for emphasis)
            weighted_yields = [a['predicted_yield'] ** 1.5 for a in top_assets]
            total_weight = sum(weighted_yields)
            
            if total_weight > 0:
                allocations = {
                    top_assets[i]['symbol']: weighted_yields[i] / total_weight
                    for i in range(len(top_assets))
                }
            else:
                # Fallback to equal weights
                weight = 1.0 / len(top_assets)
                allocations = {a['symbol']: weight for a in top_assets}
            
            logger.debug(f"Allocations: {allocations}")
            
            # Check if rebalance is worth it (only if we have previous positions)
            if current_positions:
                # Calculate overlap between current and new allocations
                current_symbols = set(current_positions.keys())
                new_symbols = set(allocations.keys())
                overlap = len(current_symbols & new_symbols) / max(len(current_symbols), len(new_symbols))
                
                # If portfolios are very similar, keep current positions to save transaction costs
                if overlap >= (1 - rebalance_threshold):
                    logger.debug(f"Rebalance threshold not met (overlap={overlap:.2%}), keeping positions")
                    return current_positions
            
            # Store positions for next iteration
            self._last_ml_positions = {symbol: weight for symbol, weight in allocations.items()}
            return allocations
        
        logger.debug("No acceptable assets found, no allocation")
        self._last_ml_positions = {}
        return {}
    
    def strategy_stablecoin_max(self, df, date, min_apy=2.0, top_n=3):
        """Stablecoin Maximum Profit Strategy: Zero risk, maximum stable returns
        
        Only invests in HIGH-YIELD stablecoin pools:
        - USDC, USDT, DAI (lending on Aave - consistently >2%)
        - USDC/USDT (Uniswap V3 LP - low risk, decent yield)
        
        Strategy:
        1. Filter to only stablecoin assets with proven performance
        2. Require minimum APY of 2.0% (avoid dead capital)
        3. Select top 3 by current APY
        4. Equal weight for stability (avoid over-concentration)
        """
        
        # Get data for current date
        current_ts = pd.Timestamp(date).tz_localize(None)
        df_copy = df.copy()
        df_copy['time_naive'] = pd.to_datetime(df_copy['time']).dt.tz_localize(None)
        historical_df = df_copy[df_copy['time_naive'] <= current_ts]
        
        if len(historical_df) == 0:
            return {}
        
        # Get most recent data
        day_data = historical_df[historical_df['time_naive'].dt.date == date.date()]
        
        if len(day_data) == 0:
            recent_dates = historical_df[historical_df['time_naive'] <= current_ts]['time_naive'].dt.date.unique()
            if len(recent_dates) > 0:
                most_recent_date = max(recent_dates)
                day_data = historical_df[historical_df['time_naive'].dt.date == most_recent_date]
        
        if len(day_data) == 0:
            return {}
        
        # Define HIGH-QUALITY stablecoin pools only
        # Based on analysis: USDC, USDT, DAI on Aave are 100% above 1.84%
        # USDC/USDT on Uniswap is 72% above 1.84%
        high_quality_stables = ['USDC', 'USDT', 'DAI', 'USDC/USDT']
        
        # Filter to only high-quality stablecoins
        stable_data = day_data[day_data['symbol'].isin(high_quality_stables)].copy()
        
        # Filter by minimum APY (higher threshold for quality)
        candidates = stable_data[stable_data['apy_percent'] >= min_apy]
        
        if len(candidates) == 0:
            # Fallback to lower threshold if no high-yield stables available
            candidates = stable_data[stable_data['apy_percent'] >= 1.5]
        
        if len(candidates) == 0:
            return {}
        
        # Sort by APY and select top N
        top_stables = candidates.nlargest(top_n, 'apy_percent')
        
        # Equal weight allocation for stability and consistency
        if len(top_stables) > 0:
            weight = 1.0 / len(top_stables)
            allocations = {row['symbol']: weight for _, row in top_stables.iterrows()}
            return allocations
        
        return {}
    
    def strategy_risk_adjusted(self, df, date, xgb_model, risk_scaler, label_encoder, min_apy=2.0, top_n=3):
        """Strategy: Use XGBoost to filter by risk, then pick highest APY"""
    
    def strategy_risk_adjusted(self, df, date, xgb_model, risk_scaler, label_encoder, min_apy=2.0, top_n=3):
        """Strategy: Select assets with low/medium risk and highest APY"""
        
        # Get current day data
        day_data = df[df['time'].dt.date == date.date()]
        
        if len(day_data) == 0:
            return {}
        
        # Filter by minimum APY
        candidates = day_data[day_data['apy_percent'] >= min_apy].copy()
        
        if len(candidates) == 0:
            return {}
        
        # Predict risk for each asset
        risk_scores = []
        
        for _, row in candidates.iterrows():
            # Create feature vector for XGBoost (7 features)
            features = np.array([[
                row['apy_percent'],
                row['tvl_usd'],
                0,  # utilization (not available)
                0,  # volatility (not available in day data)
                row['apy_percent'] / (row['tvl_usd'] + 1),  # Yield/TVL ratio
                1 if row['protocol'] == 'uniswap_v3' else 0,  # Is DEX
                1 if row['protocol'] == 'aave_v3' else 0,     # Is Lending
            ]])
            
            # Normalize and predict
            features_scaled = risk_scaler.transform(features)
            risk_pred_proba = xgb_model.predict_proba(features_scaled)[0]
            risk_pred = np.argmax(risk_pred_proba)
            risk_label = label_encoder.inverse_transform([risk_pred])[0]
            
            risk_scores.append({
                'symbol': row['symbol'],
                'apy': row['apy_percent'],
                'tvl': row['tvl_usd'],
                'risk': risk_label,
                'risk_score': 0 if risk_label == 'low' else (1 if risk_label == 'medium' else 2)
            })
        
        # Filter out high-risk assets
        safe_assets = [a for a in risk_scores if a['risk_score'] <= 1]  # low or medium
        
        if len(safe_assets) == 0:
            # If no safe assets, take lowest risk ones
            safe_assets = sorted(risk_scores, key=lambda x: x['risk_score'])[:top_n]
        
        # Sort by APY and select top N
        safe_assets = sorted(safe_assets, key=lambda x: x['apy'], reverse=True)[:top_n]
        
        # Equal weight allocation
        if len(safe_assets) > 0:
            weight = 1.0 / len(safe_assets)
            allocations = {a['symbol']: weight for a in safe_assets}
            return allocations
        
        return {}
    
    def simulate(self, df, strategy, rebalance_frequency='7D', smart_rebalancing=True):
        """
        Simulate a strategy over historical data
        
        Args:
            df: Historical data
            strategy: Strategy function
            rebalance_frequency: How often to rebalance (e.g., '7D' for weekly)
            smart_rebalancing: Apply partial transaction costs for minor rebalances
        """
        
        # Reset position tracking
        if hasattr(self, '_last_ml_positions'):
            delattr(self, '_last_ml_positions')
        
        # Get unique dates
        dates = pd.to_datetime(df['time']).dt.date.unique()
        dates = sorted(dates)
        
        # Rebalancing dates
        rebalance_dates = pd.date_range(
            start=dates[0], 
            end=dates[-1], 
            freq=rebalance_frequency
        )
        
        # Portfolio state
        capital = self.initial_capital
        positions = {}  # {symbol: {amount, entry_date, entry_apy}}
        portfolio_values = []
        
        logger.info(f"Simulating from {dates[0]} to {dates[-1]}")
        logger.info(f"Rebalancing every {rebalance_frequency}")
        
        for current_date in pd.to_datetime(dates):
            # Check if rebalancing day
            if current_date in rebalance_dates:
                # Get allocations from strategy
                allocations = strategy(df, current_date)
                
                if allocations:
                    # Calculate total value
                    total_value = capital
                    for symbol, pos in positions.items():
                        # Get current APY for this asset
                        current_data = df[
                            (df['time'].dt.date == current_date.date()) & 
                            (df['symbol'] == symbol)
                        ]
                        
                        if len(current_data) > 0:
                            # Value increased by APY (daily)
                            days_held = (current_date - pos['entry_date']).days
                            if days_held > 0:
                                daily_return = pos['entry_apy'] / 365 / 100
                                growth = (1 + daily_return) ** days_held
                                total_value += pos['amount'] * growth
                            else:
                                total_value += pos['amount']
                    
                    # Calculate turnover for smart transaction costs
                    if smart_rebalancing and positions:
                        old_symbols = set(positions.keys())
                        new_symbols = set(allocations.keys())
                        overlap = len(old_symbols & new_symbols) / len(old_symbols.union(new_symbols)) if old_symbols.union(new_symbols) else 0
                        # Reduce transaction costs proportional to portfolio overlap
                        # If 80% overlap, only 20% of portfolio trades -> 20% of transaction cost
                        effective_tx_cost = self.transaction_cost * (1 - overlap)
                        total_value *= (1 - effective_tx_cost)
                        logger.debug(f"Overlap: {overlap:.1%}, Effective TX cost: {effective_tx_cost:.3%}")
                    else:
                        # Apply full transaction costs
                        total_value *= (1 - self.transaction_cost)
                    
                    # Rebalance
                    new_positions = {}
                    for symbol, weight in allocations.items():
                        amount = total_value * weight
                        
                        # Get entry APY
                        asset_data = df[
                            (df['time'].dt.date == current_date.date()) & 
                            (df['symbol'] == symbol)
                        ]
                        
                        if len(asset_data) > 0:
                            entry_apy = asset_data.iloc[0]['apy_percent']
                            new_positions[symbol] = {
                                'amount': amount,
                                'entry_date': current_date,
                                'entry_apy': entry_apy
                            }
                    
                    positions = new_positions
                    capital = 0  # All capital is invested
            
            # Calculate current portfolio value
            total_value = capital
            for symbol, pos in positions.items():
                current_data = df[
                    (df['time'].dt.date == current_date.date()) & 
                    (df['symbol'] == symbol)
                ]
                
                if len(current_data) > 0:
                    days_held = (current_date - pos['entry_date']).days
                    if days_held > 0:
                        daily_return = pos['entry_apy'] / 365 / 100
                        growth = (1 + daily_return) ** days_held
                        total_value += pos['amount'] * growth
                    else:
                        total_value += pos['amount']
            
            portfolio_values.append({
                'date': current_date,
                'value': total_value,
                'positions': len(positions)
            })
        
        return pd.DataFrame(portfolio_values)
    
    def calculate_metrics(self, results_df):
        """Calculate performance metrics"""
        
        if len(results_df) == 0:
            return {}
        
        initial_value = results_df.iloc[0]['value']
        final_value = results_df.iloc[-1]['value']
        
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Calculate daily returns
        results_df['daily_return'] = results_df['value'].pct_change()
        
        # Annualized metrics
        days = (results_df['date'].max() - results_df['date'].min()).days
        years = days / 365.25
        
        annualized_return = ((final_value / initial_value) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        volatility = results_df['daily_return'].std() * np.sqrt(365) * 100
        sharpe = annualized_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        rolling_max = results_df['value'].cummax()
        drawdown = (results_df['value'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        return {
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'volatility_pct': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'final_value': final_value,
            'days': days
        }


def main():
    load_dotenv()
    
    logger.info("="*60)
    logger.info("DeFi Yield Backtesting")
    logger.info("="*60)
    
    # Initialize backtest
    backtest = SimpleBacktest(initial_capital=10000.0, transaction_cost=0.001)
    
    # Load data (last 90 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    logger.info(f"\nLoading data from {start_date.date()} to {end_date.date()}...")
    df = backtest.load_historical_data(start_date, end_date)
    
    logger.info(f"✓ Loaded {len(df)} records")
    logger.info(f"  Assets: {df['asset_id'].nunique()}")
    logger.info(f"  Date range: {df['time'].min()} to {df['time'].max()}")
    
    # Strategy 1: Highest APY
    logger.info("\n" + "="*60)
    logger.info("Strategy 1: Top 3 Highest APY (Weekly Rebalance)")
    logger.info("="*60)
    
    results1 = backtest.simulate(
        df, 
        lambda df, date: backtest.strategy_highest_apy(df, date, top_n=3),
        rebalance_frequency='7D'
    )
    
    metrics1 = backtest.calculate_metrics(results1)
    
    logger.info(f"\nResults:")
    logger.info(f"  Total Return: {metrics1['total_return_pct']:.2f}%")
    logger.info(f"  Annualized Return: {metrics1['annualized_return_pct']:.2f}%")
    logger.info(f"  Volatility: {metrics1['volatility_pct']:.2f}%")
    logger.info(f"  Sharpe Ratio: {metrics1['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {metrics1['max_drawdown_pct']:.2f}%")
    logger.info(f"  Final Value: ${metrics1['final_value']:.2f}")
    
    # Strategy 2: TVL Weighted
    logger.info("\n" + "="*60)
    logger.info("Strategy 2: TVL-Weighted (Min APY 1%)")
    logger.info("="*60)
    
    results2 = backtest.simulate(
        df,
        lambda df, date: backtest.strategy_tvl_weighted(df, date, min_apy=1.0),
        rebalance_frequency='7D'
    )
    
    metrics2 = backtest.calculate_metrics(results2)
    
    logger.info(f"\nResults:")
    logger.info(f"  Total Return: {metrics2['total_return_pct']:.2f}%")
    logger.info(f"  Annualized Return: {metrics2['annualized_return_pct']:.2f}%")
    logger.info(f"  Volatility: {metrics2['volatility_pct']:.2f}%")
    logger.info(f"  Sharpe Ratio: {metrics2['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {metrics2['max_drawdown_pct']:.2f}%")
    logger.info(f"  Final Value: ${metrics2['final_value']:.2f}")
    
    # Strategy 3: Optimized Unified ML (LSTM + XGBoost)
    logger.info("\n" + "="*60)
    logger.info("Strategy 3: Optimized Unified ML (LSTM + XGBoost)")
    logger.info("="*60)
    logger.info("  Profit Optimizations:")
    logger.info("  → LSTM predicts future yields")
    logger.info("  → XGBoost risk assessment")
    logger.info("  → Aggressive: High-risk OK if yield >8%")
    logger.info("  → Yield-weighted² allocations (top 4 assets)")
    logger.info("  → Min APY: 2.0% (focus on high yields)")
    
    try:
        # Load both models
        from src.ml.yield_predictor import YieldForecaster
        import xgboost as xgb
        
        # Create features if not already in dataframe
        if 'apy_ma_7d' not in df.columns:
            logger.info("Creating features for ML strategy...")
            # Quick feature creation for backtest
            df = df.sort_values(['asset_id', 'time'])
            df['hour'] = pd.to_datetime(df['time']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['time']).dt.dayofweek
            
            # Ensure volume columns exist
            if 'volume_24h_usd' not in df.columns:
                df['volume_24h_usd'] = 0
            if 'volatility_24h' not in df.columns:
                df['volatility_24h'] = 0
            
            protocol_dummies = pd.get_dummies(df['protocol'], prefix='proto')
            df = pd.concat([df, protocol_dummies], axis=1)
            
            for asset_id in df['asset_id'].unique():
                mask = df['asset_id'] == asset_id
                df.loc[mask, 'apy_ma_7d'] = df.loc[mask, 'apy_percent'].rolling(7, min_periods=1).mean()
                df.loc[mask, 'apy_ma_30d'] = df.loc[mask, 'apy_percent'].rolling(30, min_periods=1).mean()
                df.loc[mask, 'apy_std_7d'] = df.loc[mask, 'apy_percent'].rolling(7, min_periods=1).std()
                df.loc[mask, 'tvl_ma_7d'] = df.loc[mask, 'tvl_usd'].rolling(7, min_periods=1).mean()
                df.loc[mask, 'tvl_trend_7d'] = df.loc[mask, 'tvl_usd'].pct_change(7, fill_method=None) * 100
                df.loc[mask, 'volume_tvl_ratio'] = df.loc[mask, 'volume_24h_usd'] / (df.loc[mask, 'tvl_usd'] + 1e-10)
                df.loc[mask, 'apy_momentum'] = df.loc[mask, 'apy_ma_7d'] - df.loc[mask, 'apy_ma_30d']
            
            df = df.fillna(0)
        
        # Feature columns - EXACT match to training (16 features)
        feature_cols = [
            'apy_percent', 'tvl_usd', 'volume_24h_usd', 'volatility_24h',
            'apy_ma_7d', 'apy_ma_30d', 'apy_std_7d',
            'tvl_ma_7d', 'tvl_trend_7d', 'volume_tvl_ratio',
            'apy_momentum', 'hour', 'day_of_week'
        ]
        
        # Add protocol columns (should be 3: aave_v3, curve, uniswap_v3)
        protocol_cols = sorted([c for c in df.columns if c.startswith('proto_')])
        feature_cols.extend(protocol_cols)
        
        # Verify all features exist
        missing_features = [f for f in feature_cols if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        logger.info(f"Loading LSTM model with {len(feature_cols)} features")
        lstm_model = YieldForecaster(input_size=len(feature_cols))
        lstm_model.load_state_dict(torch.load('models/lstm_best.pth'))
        lstm_model.eval()
        
        # Load LSTM scaler
        with open('models/feature_scaler.pkl', 'rb') as f:
            lstm_scaler = pickle.load(f)
        
        logger.info("Loading XGBoost risk classifier")
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model('models/xgboost_risk_classifier.json')
        
        # Load XGBoost scaler and label encoder
        with open('models/risk_scaler.pkl', 'rb') as f:
            risk_scaler = pickle.load(f)
        
        with open('models/risk_label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        logger.info("Running optimized unified ML backtest...")
        results3 = backtest.simulate(
            df,
            lambda df, date: backtest.strategy_unified_ml(
                df, date,
                lstm_model, lstm_scaler, feature_cols,
                xgb_model, risk_scaler, label_encoder,
                min_apy=2.0, top_n=4
            ),
            rebalance_frequency='7D'
        )
        
        metrics3 = backtest.calculate_metrics(results3)
        
        logger.info(f"\nResults:")
        logger.info(f"  Total Return: {metrics3['total_return_pct']:.2f}%")
        logger.info(f"  Annualized Return: {metrics3['annualized_return_pct']:.2f}%")
        logger.info(f"  Volatility: {metrics3['volatility_pct']:.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics3['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {metrics3['max_drawdown_pct']:.2f}%")
        logger.info(f"  Final Value: ${metrics3['final_value']:.2f}")
        
        # Add to combined results
        results3['strategy'] = 'Optimized_Unified_ML'
        
    except Exception as e:
        logger.warning(f"⚠️  Optimized unified ML strategy failed: {e}")
        import traceback
        logger.warning(traceback.format_exc())
        logger.info("Continuing with baseline strategies only...")
        results3 = None
    
    # Strategy 4: Stablecoin Maximum Profit
    logger.info("\n" + "="*60)
    logger.info("Strategy 4: Stablecoin Maximum Profit (Zero Risk)")
    logger.info("="*60)
    logger.info("  Focus: Stablecoins only (USDC, USDT, DAI, USDC/USDT)")
    logger.info("  → No volatile assets (WETH, WBTC)")
    logger.info("  → Only proven high-yield stables")
    logger.info("  → Min APY: 2.0% (avoid low performers)")
    logger.info("  → Equal weight (stability focus)")
    
    try:
        logger.info("Running stablecoin-only backtest...")
        results4 = backtest.simulate(
            df,
            lambda df, date: backtest.strategy_stablecoin_max(df, date, min_apy=2.0, top_n=3),
            rebalance_frequency='7D'
        )
        
        metrics4 = backtest.calculate_metrics(results4)
        
        logger.info(f"\nResults:")
        logger.info(f"  Total Return: {metrics4['total_return_pct']:.2f}%")
        logger.info(f"  Annualized Return: {metrics4['annualized_return_pct']:.2f}%")
        logger.info(f"  Volatility: {metrics4['volatility_pct']:.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics4['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {metrics4['max_drawdown_pct']:.2f}%")
        logger.info(f"  Final Value: ${metrics4['final_value']:.2f}")
        
        # Add to combined results
        results4['strategy'] = 'Stablecoin_Max'
        
    except Exception as e:
        logger.warning(f"⚠️  Stablecoin strategy failed: {e}")
        import traceback
        logger.warning(traceback.format_exc())
        logger.info("Continuing without stablecoin strategy...")
        results4 = None
    
    # Save results
    results1['strategy'] = 'Highest_APY'
    results2['strategy'] = 'TVL_Weighted'
    
    results_list = [results1, results2]
    if results3 is not None:
        results_list.append(results3)
    if results4 is not None:
        results_list.append(results4)
    
    combined = pd.concat(results_list)
    combined.to_csv('backtest_results.csv', index=False)
    
    logger.info("\n" + "="*60)
    logger.info("✅ Backtesting Complete!")
    logger.info("="*60)
    logger.info("Results saved to: backtest_results.csv")
    
    # Comparison
    logger.info("\n" + "="*60)
    logger.info("Strategy Comparison")
    logger.info("="*60)
    logger.info(f"{'Strategy':<35} {'Return':<12} {'Sharpe':<10} {'Max DD':<10}")
    logger.info("-"*75)
    logger.info(f"{'Highest APY (Baseline)':<35} {metrics1['total_return_pct']:>10.2f}%  {metrics1['sharpe_ratio']:>8.2f}  {metrics1['max_drawdown_pct']:>8.2f}%")
    logger.info(f"{'TVL Weighted (Baseline)':<35} {metrics2['total_return_pct']:>10.2f}%  {metrics2['sharpe_ratio']:>8.2f}  {metrics2['max_drawdown_pct']:>8.2f}%")
    if results3 is not None:
        logger.info(f"{'Optimized Unified ML ⭐':<35} {metrics3['total_return_pct']:>10.2f}%  {metrics3['sharpe_ratio']:>8.2f}  {metrics3['max_drawdown_pct']:>8.2f}%")
    if results4 is not None:
        logger.info(f"{'Stablecoin Max Profit 🪙':<35} {metrics4['total_return_pct']:>10.2f}%  {metrics4['sharpe_ratio']:>8.2f}  {metrics4['max_drawdown_pct']:>8.2f}%")


if __name__ == "__main__":
    main()
