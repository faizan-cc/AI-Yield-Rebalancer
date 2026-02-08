"""
Feature Engineering Pipeline
Transforms raw protocol data into ML-ready feature vectors
Target: 32-dimensional feature space for LSTM and XGBoost models
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProtocolFeatures:
    """Feature vector for a single protocol at a point in time"""
    
    # Identifiers
    protocol_id: int
    protocol_name: str
    asset: str
    timestamp: datetime
    
    # Yield Features (4)
    current_apy: float
    apy_7d_ma: float  # 7-day moving average
    apy_30d_ma: float  # 30-day moving average
    apy_volatility: float  # 30-day standard deviation
    
    # Liquidity Features (4)
    total_liquidity_usd: float
    available_liquidity_usd: float
    utilization_rate: float
    liquidity_trend: float  # 7-day % change
    
    # Volume Features (3)
    volume_24h_usd: float
    volume_7d_usd: float
    volume_trend: float  # 7-day % change
    
    # Risk Features (5)
    risk_score: float  # 0-100
    exploit_count: int
    audit_score: float
    days_since_last_exploit: int
    tvl_concentration: float  # Herfindahl index
    
    # Market Features (4)
    gas_price_gwei: float
    eth_price_usd: float
    market_volatility: float  # VIX-like metric
    defi_tvl_dominance: float  # Protocol TVL / Total DeFi TVL
    
    # Time Features (4)
    day_of_week: int
    hour_of_day: int
    is_weekend: int
    days_since_epoch: int
    
    # Competitive Features (4)
    relative_apy_rank: int  # Rank among similar protocols
    apy_vs_category_avg: float  # APY - category average APY
    liquidity_vs_category_avg: float
    volume_vs_category_avg: float
    
    # Historical Performance (4)
    roi_7d: float  # 7-day return
    roi_30d: float  # 30-day return
    sharpe_ratio_30d: float
    max_drawdown_30d: float
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy array for ML models"""
        return np.array([
            # Yield (4)
            self.current_apy,
            self.apy_7d_ma,
            self.apy_30d_ma,
            self.apy_volatility,
            
            # Liquidity (4)
            np.log1p(self.total_liquidity_usd),  # Log transform
            np.log1p(self.available_liquidity_usd),
            self.utilization_rate,
            self.liquidity_trend,
            
            # Volume (3)
            np.log1p(self.volume_24h_usd),
            np.log1p(self.volume_7d_usd),
            self.volume_trend,
            
            # Risk (5)
            self.risk_score,
            self.exploit_count,
            self.audit_score,
            self.days_since_last_exploit,
            self.tvl_concentration,
            
            # Market (4)
            self.gas_price_gwei,
            np.log1p(self.eth_price_usd),
            self.market_volatility,
            self.defi_tvl_dominance,
            
            # Time (4)
            self.day_of_week,
            self.hour_of_day,
            self.is_weekend,
            self.days_since_epoch,
            
            # Competitive (4)
            self.relative_apy_rank,
            self.apy_vs_category_avg,
            self.liquidity_vs_category_avg,
            self.volume_vs_category_avg,
            
            # Historical (4)
            self.roi_7d,
            self.roi_30d,
            self.sharpe_ratio_30d,
            self.max_drawdown_30d,
        ])
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for ML models"""
        return [
            # Yield
            "current_apy", "apy_7d_ma", "apy_30d_ma", "apy_volatility",
            # Liquidity
            "log_total_liquidity", "log_available_liquidity", "utilization_rate", "liquidity_trend",
            # Volume
            "log_volume_24h", "log_volume_7d", "volume_trend",
            # Risk
            "risk_score", "exploit_count", "audit_score", "days_since_exploit", "tvl_concentration",
            # Market
            "gas_price", "log_eth_price", "market_volatility", "defi_tvl_dominance",
            # Time
            "day_of_week", "hour_of_day", "is_weekend", "days_since_epoch",
            # Competitive
            "apy_rank", "apy_vs_avg", "liquidity_vs_avg", "volume_vs_avg",
            # Historical
            "roi_7d", "roi_30d", "sharpe_30d", "max_drawdown_30d",
        ]


class FeatureEngineer:
    """
    Feature engineering pipeline for DeFi protocols
    Transforms raw data into ML-ready features
    """
    
    def __init__(self, db_connection):
        """
        Initialize feature engineer
        
        Args:
            db_connection: PostgreSQL connection
        """
        self.db = db_connection
        self.feature_cache = {}
    
    def compute_yield_features(
        self,
        protocol_id: int,
        asset: str,
        current_date: datetime,
    ) -> Dict[str, float]:
        """
        Compute yield-related features
        
        Returns:
            Dict with current_apy, apy_7d_ma, apy_30d_ma, apy_volatility
        """
        query = """
        SELECT apy_percent, recorded_at
        FROM protocol_yields
        WHERE protocol_id = %s 
          AND asset = %s
          AND recorded_at >= %s
          AND recorded_at <= %s
        ORDER BY recorded_at DESC
        """
        
        start_date = current_date - timedelta(days=30)
        
        cursor = self.db.cursor()
        cursor.execute(query, (protocol_id, asset, start_date, current_date))
        rows = cursor.fetchall()
        cursor.close()
        
        if not rows:
            return {
                "current_apy": 0.0,
                "apy_7d_ma": 0.0,
                "apy_30d_ma": 0.0,
                "apy_volatility": 0.0,
            }
        
        # Convert to pandas for easier calculations
        df = pd.DataFrame(rows, columns=["apy", "timestamp"])
        df = df.sort_values("timestamp")
        
        current_apy = df["apy"].iloc[-1] if len(df) > 0 else 0.0
        apy_7d_ma = df["apy"].tail(7).mean() if len(df) >= 7 else current_apy
        apy_30d_ma = df["apy"].mean() if len(df) > 0 else 0.0
        apy_volatility = df["apy"].std() if len(df) > 1 else 0.0
        
        return {
            "current_apy": float(current_apy),
            "apy_7d_ma": float(apy_7d_ma),
            "apy_30d_ma": float(apy_30d_ma),
            "apy_volatility": float(apy_volatility),
        }
    
    def compute_liquidity_features(
        self,
        protocol_id: int,
        asset: str,
        current_date: datetime,
    ) -> Dict[str, float]:
        """Compute liquidity-related features"""
        query = """
        SELECT total_liquidity_usd, available_liquidity_usd, 
               utilization_ratio, recorded_at
        FROM protocol_yields
        WHERE protocol_id = %s 
          AND asset = %s
          AND recorded_at >= %s
          AND recorded_at <= %s
        ORDER BY recorded_at DESC
        LIMIT 7
        """
        
        start_date = current_date - timedelta(days=7)
        
        cursor = self.db.cursor()
        cursor.execute(query, (protocol_id, asset, start_date, current_date))
        rows = cursor.fetchall()
        cursor.close()
        
        if not rows:
            return {
                "total_liquidity_usd": 0.0,
                "available_liquidity_usd": 0.0,
                "utilization_rate": 0.0,
                "liquidity_trend": 0.0,
            }
        
        latest = rows[0]
        total_liquidity = latest[0] or 0.0
        available_liquidity = latest[1] or 0.0
        utilization_rate = latest[2] or 0.0
        
        # Calculate 7-day trend
        if len(rows) >= 2:
            oldest_liquidity = rows[-1][0] or 0.0
            if oldest_liquidity > 0:
                liquidity_trend = ((total_liquidity - oldest_liquidity) / oldest_liquidity) * 100
            else:
                liquidity_trend = 0.0
        else:
            liquidity_trend = 0.0
        
        return {
            "total_liquidity_usd": float(total_liquidity),
            "available_liquidity_usd": float(available_liquidity),
            "utilization_rate": float(utilization_rate),
            "liquidity_trend": float(liquidity_trend),
        }
    
    def compute_risk_features(
        self,
        protocol_id: int,
        current_date: datetime,
    ) -> Dict[str, float]:
        """Compute risk-related features"""
        query = """
        SELECT risk_score, audit_score, exploit_count, last_exploit_date
        FROM protocol_risk_scores
        WHERE protocol_id = %s
          AND recorded_at <= %s
        ORDER BY recorded_at DESC
        LIMIT 1
        """
        
        cursor = self.db.cursor()
        cursor.execute(query, (protocol_id, current_date))
        row = cursor.fetchone()
        cursor.close()
        
        if not row:
            return {
                "risk_score": 50.0,  # Neutral risk
                "exploit_count": 0,
                "audit_score": 50.0,
                "days_since_last_exploit": 9999,
                "tvl_concentration": 0.5,
            }
        
        risk_score, audit_score, exploit_count, last_exploit = row
        
        if last_exploit:
            days_since = (current_date.date() - last_exploit).days
        else:
            days_since = 9999
        
        return {
            "risk_score": float(risk_score or 50.0),
            "exploit_count": int(exploit_count or 0),
            "audit_score": float(audit_score or 50.0),
            "days_since_last_exploit": days_since,
            "tvl_concentration": 0.5,  # Placeholder - calculate from pool distribution
        }
    
    def compute_market_features(
        self,
        current_date: datetime,
    ) -> Dict[str, float]:
        """Compute market-wide features"""
        # Get latest gas price
        gas_query = """
        SELECT fast_gwei
        FROM gas_prices
        WHERE recorded_at <= %s
        ORDER BY recorded_at DESC
        LIMIT 1
        """
        
        cursor = self.db.cursor()
        cursor.execute(gas_query, (current_date,))
        gas_row = cursor.fetchone()
        gas_price = gas_row[0] if gas_row else 50.0
        
        # Get ETH price (placeholder - would use asset_prices table)
        eth_price = 3000.0  # Placeholder
        
        cursor.close()
        
        return {
            "gas_price_gwei": float(gas_price),
            "eth_price_usd": float(eth_price),
            "market_volatility": 20.0,  # Placeholder
            "defi_tvl_dominance": 0.1,  # Placeholder
        }
    
    def compute_time_features(
        self,
        current_date: datetime,
    ) -> Dict[str, Any]:
        """Compute time-based features"""
        epoch = datetime(2024, 1, 1)
        days_since_epoch = (current_date - epoch).days
        
        return {
            "day_of_week": current_date.weekday(),
            "hour_of_day": current_date.hour,
            "is_weekend": 1 if current_date.weekday() >= 5 else 0,
            "days_since_epoch": days_since_epoch,
        }
    
    def compute_competitive_features(
        self,
        protocol_id: int,
        asset: str,
        current_apy: float,
        current_date: datetime,
    ) -> Dict[str, float]:
        """Compute competitive positioning features"""
        # Get all protocols in same category at current time
        query = """
        SELECT py.apy_percent, py.total_liquidity_usd
        FROM protocol_yields py
        JOIN protocols p ON py.protocol_id = p.id
        WHERE py.recorded_at >= %s
          AND py.recorded_at <= %s
        ORDER BY py.apy_percent DESC
        """
        
        start = current_date - timedelta(hours=1)
        
        cursor = self.db.cursor()
        cursor.execute(query, (start, current_date))
        rows = cursor.fetchall()
        cursor.close()
        
        if not rows:
            return {
                "relative_apy_rank": 1,
                "apy_vs_category_avg": 0.0,
                "liquidity_vs_category_avg": 0.0,
                "volume_vs_category_avg": 0.0,
            }
        
        apys = [r[0] for r in rows if r[0] is not None]
        avg_apy = np.mean(apys) if apys else current_apy
        
        # Rank current APY
        rank = sum(1 for apy in apys if apy > current_apy) + 1
        
        return {
            "relative_apy_rank": rank,
            "apy_vs_category_avg": current_apy - avg_apy,
            "liquidity_vs_category_avg": 0.0,  # Placeholder
            "volume_vs_category_avg": 0.0,  # Placeholder
        }
    
    def compute_historical_performance(
        self,
        protocol_id: int,
        asset: str,
        current_date: datetime,
    ) -> Dict[str, float]:
        """Compute historical return metrics"""
        query = """
        SELECT apy_percent, recorded_at
        FROM protocol_yields
        WHERE protocol_id = %s 
          AND asset = %s
          AND recorded_at >= %s
          AND recorded_at <= %s
        ORDER BY recorded_at ASC
        """
        
        start_date = current_date - timedelta(days=30)
        
        cursor = self.db.cursor()
        cursor.execute(query, (protocol_id, asset, start_date, current_date))
        rows = cursor.fetchall()
        cursor.close()
        
        if len(rows) < 2:
            return {
                "roi_7d": 0.0,
                "roi_30d": 0.0,
                "sharpe_ratio_30d": 0.0,
                "max_drawdown_30d": 0.0,
            }
        
        df = pd.DataFrame(rows, columns=["apy", "timestamp"])
        df = df.sort_values("timestamp")
        
        # Simple ROI calculation (APY changes)
        roi_7d = 0.0
        roi_30d = 0.0
        
        if len(df) >= 7:
            roi_7d = df["apy"].iloc[-1] - df["apy"].iloc[-7]
        if len(df) >= 30:
            roi_30d = df["apy"].iloc[-1] - df["apy"].iloc[0]
        
        # Sharpe ratio (return / volatility)
        returns = df["apy"].pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(365)  # Annualized
        else:
            sharpe = 0.0
        
        # Max drawdown
        cummax = df["apy"].cummax()
        drawdown = (df["apy"] - cummax) / cummax * 100
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
        
        return {
            "roi_7d": float(roi_7d),
            "roi_30d": float(roi_30d),
            "sharpe_ratio_30d": float(sharpe),
            "max_drawdown_30d": float(max_drawdown),
        }
    
    def create_feature_vector(
        self,
        protocol_id: int,
        protocol_name: str,
        asset: str,
        current_date: datetime,
    ) -> ProtocolFeatures:
        """
        Create complete feature vector for a protocol
        
        Args:
            protocol_id: Protocol ID
            protocol_name: Protocol name
            asset: Asset symbol
            current_date: Current timestamp
            
        Returns:
            ProtocolFeatures object with all 32 features
        """
        logger.info(f"Creating features for {protocol_name} {asset} at {current_date}")
        
        # Compute all feature groups
        yield_feats = self.compute_yield_features(protocol_id, asset, current_date)
        liquidity_feats = self.compute_liquidity_features(protocol_id, asset, current_date)
        risk_feats = self.compute_risk_features(protocol_id, current_date)
        market_feats = self.compute_market_features(current_date)
        time_feats = self.compute_time_features(current_date)
        competitive_feats = self.compute_competitive_features(
            protocol_id, asset, yield_feats["current_apy"], current_date
        )
        performance_feats = self.compute_historical_performance(
            protocol_id, asset, current_date
        )
        
        # Create feature object
        features = ProtocolFeatures(
            protocol_id=protocol_id,
            protocol_name=protocol_name,
            asset=asset,
            timestamp=current_date,
            
            # Yield
            current_apy=yield_feats["current_apy"],
            apy_7d_ma=yield_feats["apy_7d_ma"],
            apy_30d_ma=yield_feats["apy_30d_ma"],
            apy_volatility=yield_feats["apy_volatility"],
            
            # Liquidity
            total_liquidity_usd=liquidity_feats["total_liquidity_usd"],
            available_liquidity_usd=liquidity_feats["available_liquidity_usd"],
            utilization_rate=liquidity_feats["utilization_rate"],
            liquidity_trend=liquidity_feats["liquidity_trend"],
            
            # Volume (placeholder)
            volume_24h_usd=0.0,
            volume_7d_usd=0.0,
            volume_trend=0.0,
            
            # Risk
            risk_score=risk_feats["risk_score"],
            exploit_count=risk_feats["exploit_count"],
            audit_score=risk_feats["audit_score"],
            days_since_last_exploit=risk_feats["days_since_last_exploit"],
            tvl_concentration=risk_feats["tvl_concentration"],
            
            # Market
            gas_price_gwei=market_feats["gas_price_gwei"],
            eth_price_usd=market_feats["eth_price_usd"],
            market_volatility=market_feats["market_volatility"],
            defi_tvl_dominance=market_feats["defi_tvl_dominance"],
            
            # Time
            day_of_week=time_feats["day_of_week"],
            hour_of_day=time_feats["hour_of_day"],
            is_weekend=time_feats["is_weekend"],
            days_since_epoch=time_feats["days_since_epoch"],
            
            # Competitive
            relative_apy_rank=competitive_feats["relative_apy_rank"],
            apy_vs_category_avg=competitive_feats["apy_vs_category_avg"],
            liquidity_vs_category_avg=competitive_feats["liquidity_vs_category_avg"],
            volume_vs_category_avg=competitive_feats["volume_vs_category_avg"],
            
            # Historical
            roi_7d=performance_feats["roi_7d"],
            roi_30d=performance_feats["roi_30d"],
            sharpe_ratio_30d=performance_feats["sharpe_ratio_30d"],
            max_drawdown_30d=performance_feats["max_drawdown_30d"],
        )
        
        return features
    
    def create_training_dataset(
        self,
        start_date: datetime,
        end_date: datetime,
        protocols: List[Tuple[int, str]],
    ) -> Tuple[np.ndarray, List[ProtocolFeatures]]:
        """
        Create training dataset for ML models
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            protocols: List of (protocol_id, protocol_name) tuples
            
        Returns:
            (feature_matrix, feature_objects) tuple
        """
        logger.info(f"Creating training dataset from {start_date} to {end_date}")
        
        all_features = []
        
        # Generate daily snapshots
        current = start_date
        while current <= end_date:
            for protocol_id, protocol_name in protocols:
                # Get assets for this protocol
                cursor = self.db.cursor()
                cursor.execute("""
                    SELECT DISTINCT asset
                    FROM protocol_yields
                    WHERE protocol_id = %s
                      AND recorded_at <= %s
                    LIMIT 5
                """, (protocol_id, current))
                
                assets = [row[0] for row in cursor.fetchall()]
                cursor.close()
                
                # Create features for each asset
                for asset in assets:
                    try:
                        features = self.create_feature_vector(
                            protocol_id, protocol_name, asset, current
                        )
                        all_features.append(features)
                    except Exception as e:
                        logger.error(f"Error creating features: {e}")
            
            current += timedelta(days=1)
        
        # Convert to numpy array
        if all_features:
            feature_matrix = np.vstack([f.to_vector() for f in all_features])
            logger.info(f"Created dataset with shape {feature_matrix.shape}")
            return feature_matrix, all_features
        else:
            logger.warning("No features created")
            return np.array([]), []
