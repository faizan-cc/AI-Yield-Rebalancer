"""
RPC-based protocol data collectors
Direct on-chain queries for Aave V3 and Curve Finance
"""

import logging
from typing import Dict, List, Any
from src.data.alchemy_client import AlchemyClient

logger = logging.getLogger(__name__)


# Contract addresses
AAVE_V3_POOL = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
CURVE_REGISTRY = "0x90E00ACe148ca3b23Ac1bC8C240C2a7Dd9c2d7f5"


class AaveV3Collector:
    """Collect Aave V3 data via RPC"""
    
    # Major assets on Aave V3 Ethereum
    ASSETS = {
        "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
        "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
        "LINK": "0x514910771AF9Ca656af840dff83E8264EcF986CA",
        "UNI": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
        "AAVE": "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
    }
    
    def __init__(self, alchemy_client: AlchemyClient):
        self.client = alchemy_client
        self.pool = AAVE_V3_POOL
    
    async def get_reserve_data(self, asset_address: str) -> Dict[str, Any]:
        """
        Get reserve data for a specific asset
        Calls getReserveData(address asset) on Aave V3 Pool
        """
        # getReserveData(address) selector: 0x35ea6a75
        data = "0x35ea6a75" + asset_address[2:].zfill(64)
        
        result = await self.client.call_contract(
            to=self.pool,
            data=data
        )
        
        if result and result != "0x":
            return self._decode_reserve_data(result)
        return {}
    
    def _decode_reserve_data(self, data: str) -> Dict[str, Any]:
        """Decode Aave reserve data from hex"""
        data = data[2:]  # Remove 0x
        
        try:
            # Parse key fields (each uint256 is 64 hex chars)
            liquidity_rate = int(data[128:192], 16)
            variable_borrow_rate = int(data[192:256], 16)
            stable_borrow_rate = int(data[256:320], 16)
            
            # Convert Ray (1e27) to percentage
            return {
                "supply_apy": (liquidity_rate / 1e27) * 100,
                "variable_borrow_apy": (variable_borrow_rate / 1e27) * 100,
                "stable_borrow_apy": (stable_borrow_rate / 1e27) * 100,
            }
        except Exception as e:
            logger.error(f"Error decoding reserve data: {e}")
            return {}
    
    async def collect_all_markets(self) -> List[Dict[str, Any]]:
        """Collect data for all major Aave V3 markets"""
        results = []
        
        for symbol, address in self.ASSETS.items():
            try:
                data = await self.get_reserve_data(address)
                if data and data.get("supply_apy", 0) > 0:
                    results.append({
                        "symbol": symbol,
                        "asset_address": address,
                        "asset": symbol,  # For consistency with other collectors
                        "apy_percent": data["supply_apy"],
                        "supply_apy": data["supply_apy"],
                        "variable_borrow_apy": data["variable_borrow_apy"],
                        "stable_borrow_apy": data["stable_borrow_apy"],
                    })
                    logger.info(f"Collected Aave V3 {symbol}: {data['supply_apy']:.3f}% APY")
            except Exception as e:
                logger.error(f"Error collecting Aave {symbol}: {e}")
        
        return results


class CurveFinanceCollector:
    """Collect Curve Finance data via RPC"""
    
    # Major Curve pools
    POOLS = {
        "3pool": "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7",  # DAI/USDC/USDT
        "stETH": "0xDC24316b9AE028F1497c275EB9192a3Ea0f67022",  # ETH/stETH
        "FRAX": "0xDcEF968d416a41Cdac0ED8702fAC8128A64241A2",    # FRAX/USDC
    }
    
    def __init__(self, alchemy_client: AlchemyClient):
        self.client = alchemy_client
        self.registry = CURVE_REGISTRY
    
    async def get_pool_virtual_price(self, pool_address: str) -> float:
        """
        Get virtual price for a Curve pool
        Higher virtual price = pool earning fees
        """
        # get_virtual_price() selector: 0xbb7b8b80
        data = "0xbb7b8b80"
        
        try:
            result = await self.client.call_contract(
                to=pool_address,
                data=data
            )
            
            if result and result != "0x":
                virtual_price = int(result, 16) / 1e18
                return virtual_price
        except Exception as e:
            logger.error(f"Error getting virtual price for {pool_address}: {e}")
        
        return 1.0
    
    async def collect_all_pools(self) -> List[Dict[str, Any]]:
        """Collect data for major Curve pools"""
        results = []
        
        for name, address in self.POOLS.items():
            try:
                virtual_price = await self.get_pool_virtual_price(address)
                
                # Estimate APY from virtual price growth (simplified)
                # In production, compare with historical data
                base_apy = (virtual_price - 1.0) * 100  # Rough estimate
                
                results.append({
                    "pool_name": name,
                    "pool_address": address,
                    "asset": name,  # For consistency
                    "apy_percent": max(base_apy, 1.0),  # Placeholder
                    "virtual_price": virtual_price,
                    "base_yield": max(base_apy, 1.0),
                })
                logger.info(f"Collected Curve {name}: virtual_price={virtual_price:.4f}")
            except Exception as e:
                logger.error(f"Error collecting Curve {name}: {e}")
        
        return results
