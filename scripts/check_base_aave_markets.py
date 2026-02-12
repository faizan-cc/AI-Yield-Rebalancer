"""
Check Aave V3 Base Sepolia supported tokens and add pools
"""
import os
import sys
from pathlib import Path
from web3 import Web3
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()

# Base Sepolia test tokens (from faucet)
TOKENS = {
    'USDC': '0x036CbD53842c5426634e7929541eC2318f3dCF7e',  # USDC on Base Sepolia
    'USDbC': '0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA',  # USD Base Coin
    'DAI': '0x7683022d84F726a96c4A6611cD31DBF5409c0Ac9',   # DAI on Base Sepolia  
    'cbBTC': '0x4e77067b8c2c309BA5DD75B2e42c4f6c8E1CABBa',  # Coinbase Wrapped BTC
    'WETH': '0x4200000000000000000000000000000000000006',   # Wrapped ETH on Base
}

AAVE_POOL = "0x07eA79F68B2B3df564D0A34F8e19D9B1e339814b"

def check_base_aave_tokens():
    """Check which tokens are supported on Base Sepolia Aave V3"""
    
    print("=" * 60)
    print("AAVE V3 BASE SEPOLIA - ACTIVE MARKETS")
    print("=" * 60)
    
    # Setup Web3
    rpc_url = os.getenv('BASE_SEPOLIA_RPC_URL')
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    
    print(f"\n✅ Connected to Base Sepolia (Chain ID: {w3.eth.chain_id})")
    print(f"🏦 Aave Pool: {AAVE_POOL}\n")
    
    # Aave Pool ABI
    aave_pool_abi = [
        {
            "inputs": [{"name": "asset", "type": "address"}],
            "name": "getReserveData",
            "outputs": [{"type": "tuple", "components": [
                {"name": "configuration", "type": "uint256"},
                {"name": "liquidityIndex", "type": "uint128"},
                {"name": "currentLiquidityRate", "type": "uint128"},
                {"name": "variableBorrowIndex", "type": "uint128"},
                {"name": "currentVariableBorrowRate", "type": "uint128"},
                {"name": "currentStableBorrowRate", "type": "uint128"},
                {"name": "lastUpdateTimestamp", "type": "uint40"},
                {"name": "id", "type": "uint16"},
                {"name": "aTokenAddress", "type": "address"},
                {"name": "stableDebtTokenAddress", "type": "address"},
                {"name": "variableDebtTokenAddress", "type": "address"},
                {"name": "interestRateStrategyAddress", "type": "address"},
                {"name": "accruedToTreasury", "type": "uint128"},
                {"name": "unbacked", "type": "uint128"},
                {"name": "isolationModeTotalDebt", "type": "uint128"}
            ]}],
            "stateMutability": "view",
            "type": "function"
        }
    ]
    
    aave_pool = w3.eth.contract(address=AAVE_POOL, abi=aave_pool_abi)
    
    supported = []
    
    print(f"{'TOKEN':<10} {'STATUS':<12} {'ATOKEN ADDRESS':<45} {'APY'}")
    print("-" * 110)
    
    for symbol, address in TOKENS.items():
        try:
            reserve_data = aave_pool.functions.getReserveData(address).call()
            atoken_address = reserve_data[8]
            liquidity_rate = reserve_data[2]
            
            if atoken_address != "0x0000000000000000000000000000000000000000":
                status = "✅ ACTIVE"
                apy = (liquidity_rate / 1e25) if liquidity_rate > 0 else 0
                supported.append({
                    'symbol': symbol,
                    'address': address,
                    'atoken': atoken_address,
                    'apy': apy
                })
                print(f"{symbol:<10} {status:<12} {atoken_address:<45} {apy:.4f}%")
            else:
                status = "❌ INACTIVE"
                print(f"{symbol:<10} {status:<12} {'None':<45} {'N/A'}")
                
        except Exception as e:
            status = "❌ ERROR"
            print(f"{symbol:<10} {status:<12} {str(e)[:40]:<45}")
    
    print("\n" + "=" * 60)
    print(f"\n✅ Active Markets ({len(supported)}):")
    for token in supported:
        print(f"   • {token['symbol']}: {token['address']}")
        print(f"     aToken: {token['atoken']} (APY: {token['apy']:.4f}%)")
    
    return supported

if __name__ == '__main__':
    supported = check_base_aave_tokens()
    
    if len(supported) > 0:
        print("\n" + "=" * 60)
        print("📋 NEXT STEPS:")
        print("=" * 60)
        print("\n1. Add pools for these tokens:")
        for token in supported:
            print(f"   python scripts/add_pool_base.py --token {token['symbol']}")
        print("\n2. Get tokens from Base Sepolia faucet (you mentioned you already have some)")
        print("\n3. Deposit tokens into vault")
        print("\n4. Start keeper with Base Sepolia network")
