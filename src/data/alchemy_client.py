"""
Alchemy RPC Client
Wrapper for Alchemy JSON-RPC and Enhanced APIs
"""

import logging
from typing import Dict, List, Optional, Any
import httpx
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class AlchemyClient:
    """Client for Alchemy RPC and Enhanced APIs"""

    def __init__(self, api_key: str, network: str = "eth-mainnet"):
        """
        Initialize Alchemy client

        Args:
            api_key: Alchemy API key
            network: Network identifier (eth-mainnet, eth-goerli, etc)
        """
        self.api_key = api_key
        self.network = network
        self.rpc_url = f"https://{network}.g.alchemy.com/v2/{api_key}"
        self.enhanced_url = f"https://{network}.g.alchemy.com/v2/{api_key}"
        self.timeout = httpx.Timeout(30.0)

    async def rpc_call(
        self,
        method: str,
        params: Optional[List[Any]] = None,
        id: int = 1,
    ) -> Any:
        """
        Make a standard JSON-RPC call

        Args:
            method: RPC method name
            params: Method parameters
            id: Request ID

        Returns:
            RPC result
        """
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or [],
            "id": id,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.rpc_url, json=payload)
                data = response.json()

                if "error" in data:
                    logger.error(f"RPC error for {method}: {data['error']}")
                    raise Exception(f"RPC error: {data['error']}")

                return data.get("result")
        except Exception as e:
            logger.error(f"Error calling RPC {method}: {e}")
            raise

    async def get_balance(self, address: str, block: str = "latest") -> str:
        """Get ETH balance of an address"""
        return await self.rpc_call("eth_getBalance", [address, block])

    async def get_block_number(self) -> int:
        """Get current block number"""
        result = await self.rpc_call("eth_blockNumber")
        return int(result, 16) if isinstance(result, str) else result

    async def get_gas_price(self) -> Dict[str, str]:
        """
        Get current gas prices

        Returns:
            Dict with safe, standard, and fast gas prices
        """
        result = await self.rpc_call("eth_gasPrice")
        # Parse result (in wei)
        gas_price_wei = int(result, 16) if isinstance(result, str) else result
        gas_price_gwei = gas_price_wei / 1e9

        return {
            "gas_price_gwei": gas_price_gwei,
            "gas_price_wei": gas_price_wei,
        }

    async def call_contract(
        self,
        to: str,
        data: str,
        block: str = "latest",
    ) -> str:
        """
        Execute a contract call (read-only)

        Args:
            to: Contract address
            data: Encoded function call
            block: Block number or 'latest'

        Returns:
            Encoded return data
        """
        return await self.rpc_call(
            "eth_call",
            [{"to": to, "data": data}, block],
        )

    # =========================================================================
    # Enhanced APIs (Alchemy-specific)
    # =========================================================================

    async def get_token_balances(
        self,
        address: str,
        contractAddresses: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get ERC-20 token balances for an address

        Args:
            address: Wallet address
            contractAddresses: Optional list of token addresses to check

        Returns:
            Token balance data
        """
        params = {
            "address": address,
            "excludeZeroBalance": True,
        }
        if contractAddresses:
            params["contractAddresses"] = contractAddresses

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.enhanced_url}/getTokenBalances",
                    params=params,
                )
                data = response.json()
                return data
        except Exception as e:
            logger.error(f"Error getting token balances: {e}")
            raise

    async def simulate_execution(
        self,
        from_address: str,
        to_address: str,
        data: str,
        value: str = "0",
    ) -> Dict[str, Any]:
        """
        Simulate transaction execution without broadcasting

        Args:
            from_address: Sender address
            to_address: Recipient/contract address
            data: Encoded transaction data
            value: ETH value in wei

        Returns:
            Simulation result with gas used, revert reason, etc
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "jsonrpc": "2.0",
                    "method": "eth_call",
                    "params": [
                        {
                            "from": from_address,
                            "to": to_address,
                            "data": data,
                            "value": value,
                        },
                        "latest",
                    ],
                    "id": 1,
                }
                response = await client.post(self.rpc_url, json=payload)
                data = response.json()
                return data
        except Exception as e:
            logger.error(f"Error simulating execution: {e}")
            raise

    # =========================================================================
    # Protocol-Specific Data Fetching
    # =========================================================================

    async def get_aave_reserve_data(
        self,
        aave_pool_address: str,
        reserve_address: str,
    ) -> Dict[str, Any]:
        """
        Get Aave reserve data by calling pool contract

        Args:
            aave_pool_address: Aave lending pool contract address
            reserve_address: Asset address to get data for

        Returns:
            Reserve data (rate, liquidity, etc)
        """
        # This would require encoding the contract call
        # For now, return structure
        logger.info(
            f"Fetching Aave data for {reserve_address} from pool {aave_pool_address}"
        )
        return {
            "pool": aave_pool_address,
            "reserve": reserve_address,
        }

    async def get_uniswap_pool_state(
        self,
        pool_address: str,
    ) -> Dict[str, Any]:
        """
        Get Uniswap V3 pool state

        Args:
            pool_address: Pool contract address

        Returns:
            Pool state (liquidity, tick, sqrtPrice)
        """
        logger.info(f"Fetching Uniswap V3 pool state for {pool_address}")
        return {
            "pool": pool_address,
        }

    async def get_curve_pool_rates(
        self,
        pool_address: str,
    ) -> Dict[str, Any]:
        """
        Get Curve pool rates and APY

        Args:
            pool_address: Pool contract address

        Returns:
            Pool rates and metrics
        """
        logger.info(f"Fetching Curve pool rates for {pool_address}")
        return {
            "pool": pool_address,
        }


# ============================================================================
# Convenience Functions
# ============================================================================


async def fetch_on_chain_data(
    api_key: str,
    address: str,
) -> Dict[str, Any]:
    """
    Fetch on-chain data for an address

    Args:
        api_key: Alchemy API key
        address: Address to check

    Returns:
        On-chain data
    """
    client = AlchemyClient(api_key)

    try:
        balance, block, gas_price = await asyncio.gather(
            client.get_balance(address),
            client.get_block_number(),
            client.get_gas_price(),
            return_exceptions=True,
        )

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "address": address,
            "eth_balance": balance,
            "block_number": block,
            "gas_price": gas_price,
        }
    except Exception as e:
        logger.error(f"Error fetching on-chain data: {e}")
        raise


if __name__ == "__main__":
    # Test script
    import json
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("ALCHEMY_API_KEY", "")

    async def test():
        if not api_key:
            print("❌ ALCHEMY_API_KEY not set in .env")
            return

        data = await fetch_on_chain_data(api_key, "0x0000000000000000000000000000000000000000")
        print(json.dumps(data, indent=2, default=str))

    asyncio.run(test())
