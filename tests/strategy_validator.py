"""
Strategy Validator - Compare Testnet Performance vs Backtest Predictions

Tracks all testnet rebalances and validates against backtest expectations.
Alerts when performance deviates >20% from predictions.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from web3 import Web3
from dotenv import load_dotenv
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StrategyValidator:
    """Validates testnet performance against backtest predictions"""
    
    def __init__(self, strategy_name: str = "Optimized_ML", network: str = "sepolia"):
        """
        Initialize validator
        
        Args:
            strategy_name: Name of strategy to validate
            network: Testnet network (sepolia or base_sepolia)
        """
        self.strategy_name = strategy_name
        self.network = network
        self.backtest_results = self._load_backtest_results()
        self.testnet_results = []
        self.start_time = datetime.now()
        self.db_conn = self._connect_db()
        
        # Load Web3
        rpc_url = os.getenv(f"{network.upper()}_RPC_URL")
        self.w3 = Web3(Web3.HTTPProvider(rpc_url)) if rpc_url else None
        
        logger.info(f"✓ Validator initialized for {strategy_name} on {network}")
    
    def _connect_db(self):
        """Connect to database"""
        return psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', 5432),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
    
    def _load_backtest_results(self) -> pd.DataFrame:
        """Load backtest results from CSV"""
        backtest_file = 'backtest_results.csv'
        
        if not os.path.exists(backtest_file):
            logger.warning(f"Backtest file not found: {backtest_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(backtest_file)
        
        # Filter for this strategy
        strategy_data = df[df['strategy'] == self.strategy_name].copy()
        
        if len(strategy_data) == 0:
            logger.warning(f"No backtest data found for strategy: {self.strategy_name}")
            return pd.DataFrame()
        
        # Convert date column
        strategy_data['date'] = pd.to_datetime(strategy_data['date'])
        
        logger.info(f"✓ Loaded {len(strategy_data)} backtest records")
        return strategy_data
    
    def track_rebalance(
        self,
        tx_hash: str,
        timestamp: datetime,
        allocations: Dict[str, float],
        portfolio_value_before: float,
        portfolio_value_after: float
    ):
        """
        Track a testnet rebalance and validate against backtest
        
        Args:
            tx_hash: Transaction hash
            timestamp: Rebalance timestamp
            allocations: New allocations {asset: weight}
            portfolio_value_before: Portfolio value before rebalance
            portfolio_value_after: Portfolio value after rebalance
        """
        
        # Get transaction details
        if self.w3:
            try:
                tx = self.w3.eth.get_transaction(tx_hash)
                receipt = self.w3.eth.get_transaction_receipt(tx_hash)
                
                gas_used = receipt['gasUsed']
                gas_price = tx['gasPrice']
                gas_cost_eth = self.w3.from_wei(gas_used * gas_price, 'ether')
            except Exception as e:
                logger.error(f"Failed to fetch tx details: {e}")
                gas_used = 0
                gas_price = 0
                gas_cost_eth = 0
        else:
            gas_used = 0
            gas_price = 0
            gas_cost_eth = 0
        
        # Calculate metrics
        slippage = abs(portfolio_value_after - portfolio_value_before) / portfolio_value_before
        
        result = {
            'timestamp': timestamp,
            'tx_hash': tx_hash,
            'gas_used': gas_used,
            'gas_price_gwei': self.w3.from_wei(gas_price, 'gwei') if self.w3 else 0,
            'gas_cost_eth': float(gas_cost_eth),
            'slippage_actual': slippage,
            'portfolio_value_before': portfolio_value_before,
            'portfolio_value_after': portfolio_value_after,
            'allocations': allocations,
            'num_positions': len(allocations),
            'status': 'success' if receipt.get('status') == 1 else 'failed'
        }
        
        self.testnet_results.append(result)
        
        # Save to database
        self._save_to_db(result)
        
        # Validate against backtest
        validation = self.compare_to_backtest(result)
        
        # Log results
        logger.info(f"✓ Tracked rebalance #{len(self.testnet_results)}")
        logger.info(f"  Gas: {result['gas_cost_eth']:.6f} ETH")
        logger.info(f"  Slippage: {slippage:.4%}")
        logger.info(f"  Portfolio: ${portfolio_value_after:,.2f}")
        
        if validation['deviation_warning']:
            logger.warning(f"⚠️  DEVIATION WARNING: {validation['warning_message']}")
        
        return result, validation
    
    def compare_to_backtest(self, actual: Dict) -> Dict:
        """
        Compare actual testnet result to backtest prediction
        
        Args:
            actual: Actual testnet result
            
        Returns:
            Validation results with warnings
        """
        
        if len(self.backtest_results) == 0:
            return {
                'deviation_warning': False,
                'warning_message': 'No backtest data to compare'
            }
        
        # Find closest backtest rebalance by day index
        days_elapsed = (actual['timestamp'] - self.start_time).days
        
        # Get expected backtest values for this day
        if days_elapsed < len(self.backtest_results):
            backtest = self.backtest_results.iloc[days_elapsed]
        else:
            # Use average from last week
            backtest = self.backtest_results.tail(7).mean()
        
        # Calculate deviations
        current_return = (actual['portfolio_value_after'] - self.get_initial_value()) / self.get_initial_value()
        expected_return = (backtest.get('value', 10000) - 10000) / 10000
        return_deviation = abs(current_return - expected_return)
        
        # Check for significant deviations
        warnings = []
        
        # Gas cost check
        if actual['gas_cost_eth'] > 0.01:
            warnings.append(f"Gas cost too high: {actual['gas_cost_eth']:.6f} ETH (limit: 0.01 ETH)")
        
        # Slippage check
        if actual['slippage_actual'] > 0.005:
            warnings.append(f"Slippage too high: {actual['slippage_actual']:.4%} (limit: 0.5%)")
        
        # Return deviation check
        if return_deviation > 0.20:
            warnings.append(f"Return deviation: {return_deviation:.2%} from backtest (limit: 20%)")
        
        # Failed transaction check
        if actual['status'] == 'failed':
            warnings.append(f"Transaction FAILED: {actual['tx_hash']}")
        
        return {
            'deviation_warning': len(warnings) > 0,
            'warning_message': ' | '.join(warnings) if warnings else 'All checks passed',
            'return_deviation': return_deviation,
            'gas_check': actual['gas_cost_eth'] <= 0.01,
            'slippage_check': actual['slippage_actual'] <= 0.005,
            'status_check': actual['status'] == 'success'
        }
    
    def _save_to_db(self, result: Dict):
        """Save rebalance result to database"""
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS testnet_rebalances (
                        id SERIAL PRIMARY KEY,
                        strategy_name TEXT,
                        network TEXT,
                        timestamp TIMESTAMP,
                        tx_hash TEXT,
                        gas_used BIGINT,
                        gas_price_gwei NUMERIC,
                        gas_cost_eth NUMERIC,
                        slippage_actual NUMERIC,
                        portfolio_value_before NUMERIC,
                        portfolio_value_after NUMERIC,
                        allocations JSONB,
                        status TEXT,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                cur.execute("""
                    INSERT INTO testnet_rebalances (
                        strategy_name, network, timestamp, tx_hash,
                        gas_used, gas_price_gwei, gas_cost_eth,
                        slippage_actual, portfolio_value_before,
                        portfolio_value_after, allocations, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    self.strategy_name, self.network, result['timestamp'],
                    result['tx_hash'], result['gas_used'], result['gas_price_gwei'],
                    result['gas_cost_eth'], result['slippage_actual'],
                    result['portfolio_value_before'], result['portfolio_value_after'],
                    json.dumps(result['allocations']), result['status']
                ))
                
                self.db_conn.commit()
        except Exception as e:
            logger.error(f"Failed to save to database: {e}")
            self.db_conn.rollback()
    
    def get_initial_value(self) -> float:
        """Get initial portfolio value"""
        if len(self.testnet_results) > 0:
            return self.testnet_results[0]['portfolio_value_before']
        return 10000.0  # Default
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics from testnet results"""
        
        if len(self.testnet_results) == 0:
            return {}
        
        # Get portfolio values
        values = [r['portfolio_value_after'] for r in self.testnet_results]
        initial_value = self.get_initial_value()
        final_value = values[-1]
        
        # Calculate returns
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate daily returns
        daily_returns = np.diff(values) / values[:-1]
        
        # Time elapsed
        time_elapsed = (self.testnet_results[-1]['timestamp'] - self.testnet_results[0]['timestamp']).days
        
        # Annualized return
        if time_elapsed > 0:
            years = time_elapsed / 365.25
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
        else:
            annualized_return = 0
        
        # Volatility
        if len(daily_returns) > 1:
            volatility = np.std(daily_returns) * np.sqrt(365)
        else:
            volatility = 0
        
        # Sharpe ratio (2% risk-free rate)
        sharpe = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative = np.array(values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Gas costs
        total_gas_cost = sum(r['gas_cost_eth'] for r in self.testnet_results)
        avg_gas_cost = total_gas_cost / len(self.testnet_results)
        
        # Slippage
        avg_slippage = np.mean([r['slippage_actual'] for r in self.testnet_results])
        max_slippage = max(r['slippage_actual'] for r in self.testnet_results)
        
        # Success rate
        successful = sum(1 for r in self.testnet_results if r['status'] == 'success')
        success_rate = successful / len(self.testnet_results)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_value': final_value,
            'total_rebalances': len(self.testnet_results),
            'successful_rebalances': successful,
            'success_rate': success_rate,
            'total_gas_cost_eth': total_gas_cost,
            'avg_gas_cost_eth': avg_gas_cost,
            'avg_slippage': avg_slippage,
            'max_slippage': max_slippage,
            'days_elapsed': time_elapsed
        }
    
    def generate_report(self, output_file: str = 'testnet_validation_report.json') -> Tuple[Dict, bool]:
        """
        Generate comprehensive validation report
        
        Args:
            output_file: Path to save report
            
        Returns:
            (report dict, success boolean)
        """
        
        logger.info("Generating validation report...")
        
        # Calculate testnet metrics
        testnet_metrics = self.calculate_metrics()
        
        # Get backtest metrics for comparison
        if len(self.backtest_results) > 0:
            backtest_final = self.backtest_results.iloc[-1]
            backtest_return = (backtest_final['value'] - 10000) / 10000
            backtest_sharpe = 22.0  # From your backtest results
            backtest_max_dd = -0.0011  # -0.11% from backtest
        else:
            backtest_return = 0.079  # Default from backtest (+7.90%)
            backtest_sharpe = 22.0
            backtest_max_dd = -0.0011
        
        # Validation checks
        validations = {
            'return_match': abs(testnet_metrics.get('total_return', 0) - backtest_return) < 0.50 * abs(backtest_return),
            'sharpe_match': abs(testnet_metrics.get('sharpe_ratio', 0) - backtest_sharpe) < 0.30 * backtest_sharpe,
            'drawdown_match': testnet_metrics.get('max_drawdown', 0) > backtest_max_dd,
            'gas_cost_ok': testnet_metrics.get('avg_gas_cost_eth', 1) < 0.01,
            'slippage_ok': testnet_metrics.get('max_slippage', 1) < 0.005,
            'success_rate_ok': testnet_metrics.get('success_rate', 0) > 0.95
        }
        
        # Overall success
        success = all(validations.values())
        
        # Compile report
        report = {
            'strategy_name': self.strategy_name,
            'network': self.network,
            'generated_at': datetime.now().isoformat(),
            'testnet_metrics': testnet_metrics,
            'backtest_comparison': {
                'expected_return': backtest_return,
                'actual_return': testnet_metrics.get('total_return', 0),
                'return_deviation': abs(testnet_metrics.get('total_return', 0) - backtest_return),
                'expected_sharpe': backtest_sharpe,
                'actual_sharpe': testnet_metrics.get('sharpe_ratio', 0),
                'expected_max_dd': backtest_max_dd,
                'actual_max_dd': testnet_metrics.get('max_drawdown', 0)
            },
            'validations': validations,
            'overall_success': success,
            'recommendation': 'APPROVED for mainnet' if success else 'REJECTED - fix issues and retest'
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TESTNET VALIDATION REPORT")
        logger.info("="*60)
        logger.info(f"Strategy: {self.strategy_name}")
        logger.info(f"Network: {self.network}")
        logger.info(f"Duration: {testnet_metrics.get('days_elapsed', 0)} days")
        logger.info(f"Rebalances: {testnet_metrics.get('total_rebalances', 0)}")
        logger.info("-"*60)
        logger.info("PERFORMANCE:")
        logger.info(f"  Return: {testnet_metrics.get('total_return', 0):.2%} (expected: {backtest_return:.2%})")
        logger.info(f"  Sharpe: {testnet_metrics.get('sharpe_ratio', 0):.2f} (expected: {backtest_sharpe:.2f})")
        logger.info(f"  Max DD: {testnet_metrics.get('max_drawdown', 0):.2%} (limit: {backtest_max_dd:.2%})")
        logger.info("-"*60)
        logger.info("OPERATIONS:")
        logger.info(f"  Avg Gas: {testnet_metrics.get('avg_gas_cost_eth', 0):.6f} ETH (limit: 0.01)")
        logger.info(f"  Max Slippage: {testnet_metrics.get('max_slippage', 0):.4%} (limit: 0.5%)")
        logger.info(f"  Success Rate: {testnet_metrics.get('success_rate', 0):.1%} (target: >95%)")
        logger.info("-"*60)
        logger.info("VALIDATIONS:")
        for key, passed in validations.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"  {key}: {status}")
        logger.info("="*60)
        logger.info(f"FINAL DECISION: {'✅ APPROVED' if success else '❌ REJECTED'}")
        logger.info("="*60 + "\n")
        
        return report, success


if __name__ == "__main__":
    # Example usage
    validator = StrategyValidator("Optimized_Unified_ML", "sepolia")
    
    # Simulate tracking a rebalance
    # validator.track_rebalance(
    #     tx_hash="0x...",
    #     timestamp=datetime.now(),
    #     allocations={"USDC/aave_v3": 0.4, "WETH/uniswap_v3": 0.6},
    #     portfolio_value_before=10000,
    #     portfolio_value_after=10015
    # )
    
    # Generate report
    report, success = validator.generate_report()
    print(f"\nReport saved to: testnet_validation_report.json")
    print(f"Success: {success}")
