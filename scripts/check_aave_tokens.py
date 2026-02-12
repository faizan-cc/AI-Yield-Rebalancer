"""
Check which tokens are actually supported by Aave V3 on Sepolia
"""
import os
import sys
from pathlib import Path
from web3 import Web3
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from src.execution.contract_manager import ContractManager

load_dotenv()

# Common testnet tokens
TOKENS = {
    'USDC': '0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238',
    'DAI': '0xFF34B3d4Aee8ddCd6F9AFFFB6Fe49bD371b8a357',
    'WETH': '0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14',
    'USDT': '0xaA8E23Fb1079EA71e0a56F48a2aA51851D8433D0',
    'LINK': '0x779877A7B0D9E8603169DdbD7836e478b4624789',
    'WBTC': '0x92f3B59a79bFf5dc60c0d59eA13a44D082B2bdFC'
}

def check_aave_support():
    """Check which tokens are supported on Aave Sepolia"""
    
    print("=" * 60)
    print("AAVE V3 SEPOLIA - SUPPORTED TOKENS")
    print("=" * 60)
    
    contract_manager = ContractManager('sepolia')
    aave_adapter = contract_manager.contracts['AaveAdapter']
    w3 = contract_manager.w3
    
    aave_pool_address = aave_adapter.functions.aavePool().call()
    print(f"\n🏦 Aave Pool: {aave_pool_address}")
    print()
    
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
    
    aave_pool = w3.eth.contract(address=aave_pool_address, abi=aave_pool_abi)
    
    supported = []
    unsupported = []
    
    print(f"{'TOKEN':<8} {'STATUS':<12} {'ATOKEN ADDRESS':<45} {'LIQUIDITY RATE'}")
    print("-" * 100)
    
    for symbol, address in TOKENS.items():
        try:
            reserve_data = aave_pool.functions.getReserveData(address).call()
            atoken_address = reserve_data[8]
            liquidity_rate = reserve_data[2]
            
            if atoken_address != "0x0000000000000000000000000000000000000000":
                status = "✅ SUPPORTED"
                supported.append((symbol, address, atoken_address, liquidity_rate))
                apy = (liquidity_rate / 1e25) if liquidity_rate > 0 else 0  # Convert to percentage
                print(f"{symbol:<8} {status:<12} {atoken_address:<45} {apy:.4f}%")
            else:
                status = "❌ NOT ACTIVE"
                unsupported.append(symbol)
                print(f"{symbol:<8} {status:<12} {'None':<45} {'0.00%'}")
                
        except Exception as e:
            status = "❌ ERROR"
            unsupported.append(symbol)
            print(f"{symbol:<8} {status:<12} {str(e)[:40]:<45}")
    
    print("\n" + "=" * 60)
    print(f"\n✅ Supported Tokens ({len(supported)}):")
    for symbol, address, atoken, rate in supported:
        print(f"   • {symbol}: {address}")
        print(f"     aToken: {atoken}")
    
    print(f"\n❌ Unsupported Tokens ({len(unsupported)}):")
    for symbol in unsupported:
        print(f"   • {symbol}")
    
    print("\n" + "=" * 60)
    print("\n💡 RECOMMENDATION:")
    if len(supported) > 1:
        print(f"   Use these {len(supported)} supported tokens for multi-asset testing:")
        for symbol, _, _, _ in supported:
            print(f"   • {symbol}")
    else:
        print(f"   Only {len(supported)} token(s) available on Aave Sepolia")
        print(f"   Consider testing with USDC only, or deploy to Base Sepolia")

if __name__ == '__main__':
    check_aave_support()
