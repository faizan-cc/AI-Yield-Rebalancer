"""
ML Prediction Service
Connects trained LSTM/XGBoost models to smart contracts
Generates APY predictions and risk scores for yield pools
"""

import os
import sys
import json
import torch
import numpy as np
import xgboost as xgb
import pickle
from web3 import Web3
from eth_account import Account
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Import existing contract manager
from contract_manager import ContractManager

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseLogger:
    """Log predictions and rebalancing to PostgreSQL"""
    
    def __init__(self, db_name: str = "defi_yield_db"):
        """Initialize database connection"""
        try:
            # Use peer authentication (no password needed for local connections)
            self.conn = psycopg2.connect(
                dbname=db_name,
                user=os.getenv('DB_USER', os.getenv('USER', 'faizan'))
            )
            logger.info(f"✓ Connected to database: {db_name}")
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            self.conn = None
    
    def log_prediction(self, prediction: Dict) -> bool:
        """Log ML prediction to database"""
        if not self.conn:
            return False
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ml_predictions (
                        network, pool_address, asset_address, protocol_name,
                        predicted_apy, risk_level, confidence_score,
                        model_version, lstm_prediction, xgboost_risk_score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING prediction_id
                """, (
                    prediction.get('network', 'sepolia'),
                    prediction['pool_address'],
                    prediction['asset_address'],
                    prediction.get('protocol_name', 'Unknown'),
                    prediction['predicted_apy'],
                    prediction['risk_level'],
                    prediction['confidence'],
                    'v1.0',  # model_version
                    prediction['predicted_apy'],  # lstm_prediction
                    prediction['confidence']  # xgboost_risk_score
                ))
                
                prediction_id = cur.fetchone()[0]
                self.conn.commit()
                logger.info(f"✓ Logged prediction #{prediction_id} to database")
                return True
                
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
            self.conn.rollback()
            return False
    
    def log_rebalance(self, rebalance_data: Dict) -> bool:
        """Log rebalancing event to database"""
        if not self.conn:
            return False
        
        try:
            with self.conn.cursor() as cur:
                # Insert rebalance history
                cur.execute("""
                    INSERT INTO rebalance_history (
                        network, vault_address, asset_address, total_assets,
                        tx_hash, gas_used, gas_price, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING rebalance_id
                """, (
                    rebalance_data['network'],
                    rebalance_data['vault_address'],
                    rebalance_data['asset_address'],
                    rebalance_data['total_assets'],
                    rebalance_data['tx_hash'],
                    rebalance_data['gas_used'],
                    rebalance_data['gas_price'],
                    rebalance_data['status']
                ))
                
                rebalance_id = cur.fetchone()[0]
                
                # Insert pool allocations
                for pool_addr, allocation in rebalance_data['allocations'].items():
                    cur.execute("""
                        INSERT INTO pool_allocations (
                            rebalance_id, pool_address, allocation_percentage,
                            allocated_amount, pool_apy, predicted_apy, risk_level
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        rebalance_id,
                        pool_addr,
                        allocation['percentage'],
                        allocation['amount'],
                        allocation.get('pool_apy', 0),
                        allocation.get('predicted_apy', 0),
                        allocation.get('risk_level', 'medium')
                    ))
                
                self.conn.commit()
                logger.info(f"✓ Logged rebalance #{rebalance_id} to database")
                return True
                
        except Exception as e:
            logger.error(f"Failed to log rebalance: {e}")
            self.conn.rollback()
            return False
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class LSTMPredictor:
    """LSTM model for APY prediction"""
    
    def __init__(self, model_path: str):
        """Load trained LSTM model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model state dict
        logger.info(f"Loading LSTM model from {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Load feature scaler
        scaler_path = 'models/feature_scaler.pkl'
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("Feature scaler loaded")
        else:
            logger.warning("Feature scaler not found - using raw features")
            self.scaler = None
    
    def _load_model(self, model_path: str):
        """Load LSTM model architecture and weights"""
        # Model architecture matching the saved checkpoint
        class YieldLSTM(torch.nn.Module):
            def __init__(self, input_dim=18, hidden_dim=128):
                super().__init__()
                
                # LSTM layers (matching saved checkpoint)
                self.lstm1 = torch.nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=2,
                    batch_first=True,
                    bidirectional=False,
                    dropout=0.3
                )
                
                self.lstm2 = torch.nn.LSTM(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim // 2,
                    num_layers=1,
                    batch_first=True,
                    dropout=0
                )
                
                # Attention mechanism
                self.attention = torch.nn.MultiheadAttention(
                    embed_dim=hidden_dim // 2,
                    num_heads=4,
                    batch_first=True
                )
                
                # Final FC layer
                self.fc = torch.nn.Linear(hidden_dim // 2, 1)
                
            def forward(self, x):
                # LSTM layers
                lstm_out1, _ = self.lstm1(x)
                lstm_out2, _ = self.lstm2(lstm_out1)
                
                # Attention
                attn_out, _ = self.attention(lstm_out2, lstm_out2, lstm_out2)
                
                # Take last timestep
                last_hidden = attn_out[:, -1, :]
                
                # Output
                output = self.fc(last_hidden)
                
                return output
        
        model = YieldLSTM()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model
    
    def predict(self, features: np.ndarray) -> float:
        """
        Predict APY for given features
        
        Args:
            features: Feature array (sequence_length, num_features)
            
        Returns:
            Predicted APY as percentage (e.g., 5.23 for 5.23%)
        """
        try:
            # Scale features if scaler available
            if self.scaler is not None:
                features = self.scaler.transform(features.reshape(-1, features.shape[-1]))
                features = features.reshape(1, -1, features.shape[-1])
            else:
                features = features.reshape(1, -1, features.shape[-1])
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).to(self.device)
            
            # Predict
            with torch.no_grad():
                prediction = self.model(features_tensor)
                apy = prediction.item()
            
            # Ensure positive APY
            apy = max(0.0, apy)
            
            logger.info(f"LSTM predicted APY: {apy:.4f}%")
            return apy
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return 0.0


class RiskClassifier:
    """XGBoost model for risk classification"""
    
    def __init__(self, model_path: str):
        """Load trained XGBoost model"""
        logger.info(f"Loading XGBoost model from {model_path}")
        
        # Load XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        # Load risk scaler and label encoder
        scaler_path = 'models/risk_scaler.pkl'
        encoder_path = 'models/risk_label_encoder.pkl'
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("Risk scaler loaded")
        else:
            self.scaler = None
            
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            logger.info("Label encoder loaded")
        else:
            # Default risk levels
            self.label_encoder = None
            
    def predict_risk_score(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict risk level and confidence
        
        Args:
            features: Feature array (num_features,)
            
        Returns:
            Tuple of (risk_level, confidence_score)
            risk_level: 'low', 'medium', 'high'
            confidence_score: 0-100
        """
        try:
            # Scale features
            if self.scaler is not None:
                features = self.scaler.transform(features.reshape(1, -1))
            else:
                features = features.reshape(1, -1)
            
            # Create DMatrix for prediction
            dmatrix = xgb.DMatrix(features)
            
            # Predict probabilities
            probs = self.model.predict(dmatrix)[0]
            
            # Get predicted class and confidence
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class]) * 100
            
            # Map to risk level
            if self.label_encoder is not None:
                risk_level = self.label_encoder.inverse_transform([predicted_class])[0]
            else:
                risk_levels = ['low', 'medium', 'high']
                risk_level = risk_levels[min(predicted_class, 2)]
            
            logger.info(f"Risk prediction: {risk_level} (confidence: {confidence:.2f}%)")
            return risk_level, confidence
            
        except Exception as e:
            logger.error(f"Risk prediction failed: {e}")
            return 'medium', 50.0


class MLPredictionService:
    """Main service for ML-driven pool predictions"""
    
    def __init__(self, network: str = "base_sepolia"):
        """
        Initialize ML prediction service
        
        Args:
            network: Network to operate on (sepolia, base_sepolia, etc.)
        """
        self.network = network
        self.contract_manager = ContractManager(network)
        
        # Load ML models
        lstm_path = 'models/lstm_yield_predictor.pth'
        xgb_path = 'models/xgboost_risk_classifier.json'
        
        self.lstm_predictor = LSTMPredictor(lstm_path)
        self.risk_classifier = RiskClassifier(xgb_path)
        
        # Initialize database logger
        self.db_logger = DatabaseLogger()
        
        logger.info(f"ML Prediction Service initialized for {network}")
    
    def get_pool_features(self, pool_address: str, asset_address: str) -> Dict:
        """
        Fetch current pool features for prediction
        
        Args:
            pool_address: Address of the yield protocol adapter
            asset_address: Address of the asset token
            
        Returns:
            Dictionary of features
        """
        try:
            # Get current pool state from contract
            strategy_manager = self.contract_manager.contracts.get('StrategyManager')
            
            if strategy_manager:
                # Calculate poolId (keccak256 of asset + protocol)
                pool_id = self.contract_manager.w3.keccak(
                    self.contract_manager.w3.to_bytes(hexstr=asset_address) +
                    self.contract_manager.w3.to_bytes(hexstr=pool_address)
                )
                
                # Get pool info
                pool_info = strategy_manager.functions.getPool(pool_id).call()
                
                # Pool struct: (protocol, protocolName, token, currentAPY, tvl, riskScore, lastUpdate, isActive)
                current_apy = pool_info[3] / 100  # Convert from basis points
                tvl = pool_info[4] / 1e18 if pool_info[4] > 0 else 0  # Convert from wei
                
                logger.info(f"Pool {pool_address[:10]}... current APY: {current_apy}%, TVL: ${tvl:,.2f}")
            else:
                current_apy = 0.0
                tvl = 0.0
            
            # Build feature dictionary
            features = {
                'current_apy': current_apy,
                'tvl': tvl,
                'timestamp': datetime.now().timestamp(),
                'pool_address': pool_address,
                'asset_address': asset_address
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to fetch pool features: {e}")
            return {}
    
    def generate_prediction(self, pool_address: str, asset_address: str) -> Dict:
        """
        Generate ML prediction for a pool
        
        Args:
            pool_address: Address of the yield protocol adapter
            asset_address: Address of the asset token
            
        Returns:
            Dictionary with predicted_apy, risk_level, confidence
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating prediction for pool: {pool_address[:10]}...")
        logger.info(f"Asset: {asset_address[:10]}...")
        
        # Get pool features
        features = self.get_pool_features(pool_address, asset_address)
        
        if not features:
            logger.warning("No features available - using default prediction")
            return {
                'predicted_apy': 5.0,
                'risk_level': 'medium',
                'confidence': 50.0,
                'timestamp': datetime.now().isoformat()
            }
        
        # Prepare features for LSTM (use sequence of length 1 with 18 features)
        # In production, you'd use historical sequence data
        lstm_features = np.array([
            features.get('current_apy', 0),
            features.get('tvl', 0),
            features.get('timestamp', 0) % 86400,  # Time of day
            0,  # Historical mean (placeholder)
            0,  # Historical std (placeholder)
            0,  # Volume (placeholder)
            0,  # Liquidity depth (placeholder)
            0,  # Utilization rate (placeholder)
            0,  # Reserve ratio (placeholder)
            0,  # Price impact (placeholder)
            0,  # apy_7d_ma
            0,  # apy_30d_ma
            0,  # tvl_7d_ma
            0,  # volatility_7d
            0,  # apy_trend_7d
            0,  # tvl_pct_change_7d
            0,  # hour_sin
            0   # hour_cos
        ])
        
        # Reshape for LSTM: (sequence_length=7, features=18)
        lstm_input = np.tile(lstm_features, (7, 1))
        
        # Predict APY
        predicted_apy = self.lstm_predictor.predict(lstm_input)
        
        # Prepare features for risk classification (7 features to match scaler)
        risk_features = np.array([
            predicted_apy,
            features.get('tvl', 0),
            features.get('current_apy', 0),
            abs(predicted_apy - features.get('current_apy', 0)),  # APY volatility
            0,  # Default rate (placeholder)
            0,  # Audit score (placeholder)
            0   # Additional risk metric (placeholder)
        ])
        
        # Predict risk
        risk_level, confidence = self.risk_classifier.predict_risk_score(risk_features)
        
        prediction = {
            'predicted_apy': round(predicted_apy, 4),
            'risk_level': risk_level,
            'confidence': round(confidence, 2),
            'timestamp': datetime.now().isoformat(),
            'pool_address': pool_address,
            'asset_address': asset_address,
            'network': self.network
        }
        
        # Log to database
        self.db_logger.log_prediction(prediction)
        
        logger.info(f"Prediction complete: APY={predicted_apy:.4f}%, Risk={risk_level}, Confidence={confidence:.2f}%")
        logger.info(f"{'='*60}\n")
        
        return prediction
    
    def update_pool_predictions(self, pools: List[Tuple[str, str]]) -> bool:
        """
        Generate predictions and update StrategyManager contract
        
        Args:
            pools: List of (pool_address, asset_address) tuples
            
        Returns:
            True if update successful
        """
        logger.info(f"\n{'='*60}")
        logger.info("Starting ML Prediction Update")
        logger.info(f"Updating {len(pools)} pools")
        logger.info(f"{'='*60}\n")
        
        predictions = []
        
        # Generate predictions for all pools
        for pool_address, asset_address in pools:
            prediction = self.generate_prediction(pool_address, asset_address)
            predictions.append(prediction)
        
        # Update contract with new APYs
        try:
            strategy_manager = self.contract_manager.contracts.get('StrategyManager')
            
            if not strategy_manager:
                logger.error("StrategyManager contract not available")
                return False
            
            # Prepare transaction data
            pool_ids = []
            predicted_apys = []
            tvls = []
            risk_scores = []
            
            for pred in predictions:
                # Calculate poolId
                pool_id = self.contract_manager.w3.keccak(
                    self.contract_manager.w3.to_bytes(hexstr=pred['asset_address']) +
                    self.contract_manager.w3.to_bytes(hexstr=pred['pool_address'])
                )
                pool_ids.append(pool_id)
                
                # Convert APY to basis points (multiply by 100)
                predicted_apys.append(int(pred['predicted_apy'] * 100))
                
                # Use 0 TVL (we don't have real TVL data yet)
                tvls.append(0)
                
                # Convert risk level to score (low=25, medium=50, high=75)
                risk_map = {'low': 25, 'medium': 50, 'high': 75}
                risk_scores.append(risk_map.get(pred['risk_level'], 50))
            
            logger.info(f"\nUpdating on-chain pool data:")
            for i, pred in enumerate(predictions):
                logger.info(f"  Pool {i+1}: {pred['pool_address'][:10]}... → APY={pred['predicted_apy']:.4f}%, Risk={pred['risk_level']}")
            
            # Get current gas price and add 20% buffer to avoid underpriced errors
            base_gas_price = self.contract_manager.w3.eth.gas_price
            gas_price = int(base_gas_price * 1.2)
            
            # Get nonce including pending transactions
            nonce = self.contract_manager.w3.eth.get_transaction_count(
                self.contract_manager.account.address,
                'pending'  # Include pending transactions
            )
            
            logger.info(f"Gas price: {self.contract_manager.w3.from_wei(gas_price, 'gwei'):.2f} gwei")
            logger.info(f"Nonce: {nonce}")
            
            # Build transaction
            tx = strategy_manager.functions.batchUpdatePools(
                pool_ids,
                predicted_apys,
                tvls,
                risk_scores
            ).build_transaction({
                'from': self.contract_manager.account.address,
                'nonce': nonce,
                'gas': 500000,
                'gasPrice': gas_price
            })
            
            # Sign and send
            signed_tx = self.contract_manager.account.sign_transaction(tx)
            tx_hash = self.contract_manager.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            logger.info(f"\nTransaction sent: {tx_hash.hex()}")
            logger.info("Waiting for confirmation...")
            
            # Wait for confirmation
            receipt = self.contract_manager.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
            
            if receipt['status'] == 1:
                logger.info("✅ Pool update successful!")
                logger.info(f"Gas used: {receipt['gasUsed']:,}")
                logger.info(f"Transaction: https://{self.network}.etherscan.io/tx/{tx_hash.hex()}")
                logger.info(f"\n{'='*60}\n")
                return True
            else:
                logger.error("❌ Transaction failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update pools: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_optimal_allocation(self, asset_address: str, pools: List[str]) -> Dict:
        """
        Get ML-recommended optimal allocation across pools
        
        Args:
            asset_address: Asset to allocate
            pools: List of pool addresses
            
        Returns:
            Dictionary mapping pool_address to allocation percentage
        """
        logger.info(f"\nCalculating optimal allocation for {len(pools)} pools")
        
        predictions = []
        for pool in pools:
            pred = self.generate_prediction(pool, asset_address)
            predictions.append(pred)
        
        # Risk-adjusted return calculation
        scores = []
        for pred in predictions:
            # Risk weights: low=1.0, medium=0.7, high=0.4
            risk_weight = {'low': 1.0, 'medium': 0.7, 'high': 0.4}.get(pred['risk_level'], 0.5)
            
            # Score = APY * risk_weight * confidence
            score = pred['predicted_apy'] * risk_weight * (pred['confidence'] / 100)
            scores.append(score)
        
        # Normalize to percentages
        total_score = sum(scores)
        if total_score == 0:
            # Equal allocation if all scores are zero
            allocations = {pool: 100 // len(pools) for pool in pools}
        else:
            allocations = {
                pools[i]: int((scores[i] / total_score) * 100)
                for i in range(len(pools))
            }
        
        # Ensure total is 100%
        total_alloc = sum(allocations.values())
        if total_alloc != 100:
            # Add difference to highest scoring pool
            max_pool = max(allocations, key=allocations.get)
            allocations[max_pool] += (100 - total_alloc)
        
        logger.info("\nOptimal Allocation:")
        for pool, alloc in allocations.items():
            logger.info(f"  {pool[:10]}... → {alloc}%")
        
        return allocations


def main():
    """Test ML prediction service"""
    print("="*60)
    print("ML Prediction Service - Test Run")
    print("="*60)
    
    # Initialize service
    service = MLPredictionService(network='sepolia')
    
    # Load deployment info
    with open('deployments/sepolia_deployment.json', 'r') as f:
        deployment = json.load(f)
    
    # Get pool addresses
    aave_adapter = deployment['contracts']['AaveAdapter']
    usdc_address = "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238"  # Faucet USDC
    
    # Test single prediction
    print("\n" + "="*60)
    print("TEST 1: Single Pool Prediction")
    print("="*60)
    prediction = service.generate_prediction(aave_adapter, usdc_address)
    print(f"\nPrediction Results:")
    print(f"  Predicted APY: {prediction['predicted_apy']}%")
    print(f"  Risk Level: {prediction['risk_level']}")
    print(f"  Confidence: {prediction['confidence']}%")
    
    # Test pool update
    print("\n" + "="*60)
    print("TEST 2: Update Pool APYs On-Chain")
    print("="*60)
    pools = [(aave_adapter, usdc_address)]
    success = service.update_pool_predictions(pools)
    
    if success:
        print("\n✅ ML prediction service working correctly!")
    else:
        print("\n❌ ML prediction update failed")
    
    # Test optimal allocation
    print("\n" + "="*60)
    print("TEST 3: Optimal Allocation Calculation")
    print("="*60)
    allocation = service.get_optimal_allocation(usdc_address, [aave_adapter])
    print(f"\nRecommended Allocation:")
    for pool, pct in allocation.items():
        print(f"  {pool[:10]}... → {pct}%")
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
