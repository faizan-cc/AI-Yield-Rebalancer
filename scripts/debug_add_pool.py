"""
Debug why addPool transaction is failing
"""
import os
import sys
from pathlib import Path
from web3 import Web3
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from src.execution.contract_manager import ContractManager

load_dotenv()

WETH_ADDRESS = "0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14"

def debug_add_pool():
    """Debug the addPool transaction failure"""
    
    print("=" * 60)
    print("DEBUG: ADDPOOL TRANSACTION FAILURE")
    print("=" * 60)
    
    contract_manager = ContractManager('sepolia')
    strategy_manager = contract_manager.contracts['StrategyManager']
    aave_adapter = contract_manager.contracts['AaveAdapter']
    vault = contract_manager.contracts['YieldVault']
    account = contract_manager.account
    w3 = contract_manager.w3
    
    print(f"\n📊 Contract Addresses:")
    print(f"   StrategyManager: {strategy_manager.address}")
    print(f"   AaveAdapter: {aave_adapter.address}")
    print(f"   YieldVault: {vault.address}")
    print(f"   Wallet: {account.address}")
    
    # 1. Check ownership
    print(f"\n🔐 Ownership Check:")
    sm_owner = strategy_manager.functions.owner().call()
    adapter_owner = aave_adapter.functions.owner().call()
    print(f"   StrategyManager owner: {sm_owner}")
    print(f"   AaveAdapter owner: {adapter_owner}")
    print(f"   Your address: {account.address}")
    print(f"   You own StrategyManager: {sm_owner.lower() == account.address.lower()}")
    print(f"   You own AaveAdapter: {adapter_owner.lower() == account.address.lower()}")
    
    # 2. Check if pool already exists
    print(f"\n🔍 Pool Existence Check:")
    pool_id = Web3.solidity_keccak(['address', 'address'], [WETH_ADDRESS, aave_adapter.address])
    print(f"   Pool ID (WETH+Aave): {pool_id.hex()}")
    
    try:
        pool_data = strategy_manager.functions.getPool(pool_id).call()
        pool_exists = pool_data[0] != "0x0000000000000000000000000000000000000000"
        print(f"   Pool exists: {pool_exists}")
        if pool_exists:
            print(f"   Pool data: {pool_data}")
            print(f"   ❌ Pool already registered!")
            return
    except Exception as e:
        print(f"   Error checking pool: {e}")
    
    # 3. Check if WETH is supported by Aave on Sepolia
    print(f"\n🏦 Aave WETH Support Check:")
    aave_pool_address = aave_adapter.functions.aavePool().call()
    print(f"   Aave Pool Address: {aave_pool_address}")
    
    # Try to get WETH reserve data from Aave
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
    
    try:
        reserve_data = aave_pool.functions.getReserveData(WETH_ADDRESS).call()
        atoken_address = reserve_data[8]  # aTokenAddress is at index 8
        print(f"   ✅ WETH is supported by Aave!")
        print(f"   aWETH Address: {atoken_address}")
        print(f"   Liquidity Rate: {reserve_data[2]}")
    except Exception as e:
        print(f"   ❌ WETH not supported by Aave on Sepolia: {e}")
        print(f"   This is likely why addPool is failing!")
        return
    
    # 4. Check if AaveAdapter can get the aToken
    print(f"\n🔗 AaveAdapter WETH Check:")
    try:
        weth_atoken = aave_adapter.functions.getAToken(WETH_ADDRESS).call()
        print(f"   AaveAdapter can fetch aToken: {weth_atoken}")
        if weth_atoken == "0x0000000000000000000000000000000000000000":
            print(f"   ❌ AaveAdapter returns null aToken for WETH")
        else:
            print(f"   ✅ AaveAdapter recognizes WETH")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 5. Try to simulate the addPool call
    print(f"\n🧪 Simulate addPool Call:")
    try:
        # Call (not send) to see if it would revert
        result = strategy_manager.functions.addPool(
            WETH_ADDRESS,
            aave_adapter.address,
            "AaveV3"
        ).call({'from': account.address})
        print(f"   ✅ Simulation successful - transaction should work!")
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   ❌ Simulation failed: {e}")
        error_msg = str(e)
        
        # Parse common revert reasons
        if "Pool already exists" in error_msg:
            print(f"   Reason: Pool is already registered")
        elif "Invalid token" in error_msg:
            print(f"   Reason: Token address is invalid")
        elif "Invalid protocol" in error_msg:
            print(f"   Reason: Protocol address is invalid")
        elif "Ownable: caller is not the owner" in error_msg:
            print(f"   Reason: Not the contract owner")
        else:
            print(f"   Reason: Unknown - check contract requirements")
    
    # 6. Check existing pools
    print(f"\n📋 Existing Pools:")
    try:
        pool_ids = strategy_manager.functions.getPoolIds().call()
        print(f"   Total pools: {len(pool_ids)}")
        for i, pid in enumerate(pool_ids[:5]):  # Show first 5
            try:
                pool = strategy_manager.functions.getPool(pid).call()
                print(f"   Pool {i}: {pool[0][:10]}...{pool[0][-6:]} via {pool[1][:10]}...{pool[1][-6:]} ({pool[2]})")
            except:
                pass
    except Exception as e:
        print(f"   Could not fetch pools: {e}")
    
    # 7. Check YieldVault relationship
    print(f"\n🏛️ YieldVault Check:")
    try:
        vault_strategy_manager = vault.functions.strategyManager().call()
        print(f"   Vault's StrategyManager: {vault_strategy_manager}")
        print(f"   Matches: {vault_strategy_manager.lower() == strategy_manager.address.lower()}")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == '__main__':
    debug_add_pool()
