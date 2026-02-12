"""
Add correct USDC pool (0xba50Cd2A20f6DA35D788639E581bca8d0B5d4D5f) to Base Sepolia
"""
import os
import sys
from pathlib import Path
from web3 import Web3
from dotenv import load_dotenv
import json

sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()

# Correct USDC address on Base Sepolia
USDC_ADDRESS = "0xba50Cd2A20f6DA35D788639E581bca8d0B5d4D5f"
AAVE_POOL = "0x07eA79F68B2B3df564D0A34F8e19D9B1e339814b"

def check_and_add_usdc_pool():
    """Check if USDC is supported on Aave and add pool"""
    
    print("=" * 60)
    print("ADD CORRECT USDC POOL TO BASE SEPOLIA")
    print("=" * 60)
    
    # Setup Web3
    rpc_url = os.getenv('BASE_SEPOLIA_RPC_URL')
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    
    print(f"\n✅ Connected to Base Sepolia (Chain ID: {w3.eth.chain_id})")
    print(f"💵 USDC Address: {USDC_ADDRESS}")
    
    # Get account
    private_key = os.getenv('DEPLOYER_PRIVATE_KEY')
    account = w3.eth.account.from_key(private_key)
    
    # Check USDC balance
    erc20_abi = [
        {"constant": True, "inputs": [{"name": "_owner", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}], "type": "function"},
        {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"}
    ]
    usdc_contract = w3.eth.contract(address=USDC_ADDRESS, abi=erc20_abi)
    balance = usdc_contract.functions.balanceOf(account.address).call()
    print(f"💰 Your USDC Balance: {balance / (10**6)} USDC")
    
    # Check if supported by Aave
    print(f"\n🔍 Checking Aave support...")
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
    
    try:
        reserve_data = aave_pool.functions.getReserveData(USDC_ADDRESS).call()
        atoken_address = reserve_data[8]
        liquidity_rate = reserve_data[2]
        
        if atoken_address != "0x0000000000000000000000000000000000000000":
            apy = (liquidity_rate / 1e25) if liquidity_rate > 0 else 0
            print(f"✅ USDC is supported by Aave!")
            print(f"   aToken: {atoken_address}")
            print(f"   APY: {apy:.4f}%")
        else:
            print(f"❌ USDC not active on Aave (null aToken)")
            return
    except Exception as e:
        print(f"❌ Error checking Aave: {e}")
        return
    
    # Load deployment
    with open('deployments/base_sepolia_deployment.json', 'r') as f:
        deployment = json.load(f)
    
    strategy_manager_address = deployment['contracts']['StrategyManager']
    aave_adapter_address = deployment['contracts']['AaveAdapter']
    
    print(f"\n📋 StrategyManager: {strategy_manager_address}")
    print(f"🔧 AaveAdapter: {aave_adapter_address}")
    
    # Load ABI
    with open('artifacts/contracts/strategies/StrategyManager.sol/StrategyManager.json', 'r') as f:
        strategy_manager_abi = json.load(f)['abi']
    
    strategy_manager = w3.eth.contract(
        address=strategy_manager_address,
        abi=strategy_manager_abi
    )
    
    # Check if pool exists
    pool_id = Web3.solidity_keccak(['address', 'address'], [USDC_ADDRESS, aave_adapter_address])
    print(f"\n🔍 Pool ID: {pool_id.hex()}")
    
    try:
        pool_data = strategy_manager.functions.getPool(pool_id).call()
        if pool_data[0] != "0x0000000000000000000000000000000000000000":
            print(f"✅ USDC pool already exists!")
            print(f"   Token: {pool_data[0]}")
            print(f"   Protocol: {pool_data[1]}")
            return
    except:
        pass
    
    # Add pool
    print(f"\n🔄 Adding USDC pool...")
    
    try:
        gas_price = w3.eth.gas_price
        gas_price_buffered = int(gas_price * 1.2)
        nonce = w3.eth.get_transaction_count(account.address, 'pending')
        
        transaction = strategy_manager.functions.addPool(
            USDC_ADDRESS,
            aave_adapter_address,
            "AaveV3"
        ).build_transaction({
            'from': account.address,
            'gas': 250000,
            'gasPrice': gas_price_buffered,
            'nonce': nonce,
            'chainId': w3.eth.chain_id
        })
        
        signed_txn = account.sign_transaction(transaction)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        print(f"📤 Transaction: {tx_hash.hex()}")
        
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        
        if receipt['status'] == 1:
            print(f"✅ USDC pool added! Gas used: {receipt['gasUsed']}")
            print(f"\n🎉 Ready to deposit USDC and start keeper!")
        else:
            print(f"❌ Transaction failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    check_and_add_usdc_pool()
