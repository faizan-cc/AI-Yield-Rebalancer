"""
Add USDC and WETH pools to Base Sepolia StrategyManager
"""
import os
import sys
from pathlib import Path
from web3 import Web3
from dotenv import load_dotenv
import json

sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()

# Base Sepolia tokens
USDC_ADDRESS = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"
WETH_ADDRESS = "0x4200000000000000000000000000000000000006"

def add_pools_base():
    """Add USDC and WETH pools on Base Sepolia"""
    
    print("=" * 60)
    print("ADD POOLS TO BASE SEPOLIA STRATEGY MANAGER")
    print("=" * 60)
    
    # Load deployment info
    deployment_file = "deployments/base_sepolia_deployment.json"
    with open(deployment_file, 'r') as f:
        deployment = json.load(f)
    
    # Setup Web3
    rpc_url = os.getenv('BASE_SEPOLIA_RPC_URL')
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    
    print(f"\n✅ Connected to Base Sepolia (Chain ID: {w3.eth.chain_id})")
    
    # Get account
    private_key = os.getenv('DEPLOYER_PRIVATE_KEY')
    account = w3.eth.account.from_key(private_key)
    
    print(f"📍 Deployer: {account.address}")
    print(f"💰 Balance: {w3.from_wei(w3.eth.get_balance(account.address), 'ether')} ETH")
    
    # Load contracts
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
    
    # Add pools
    pools = [
        {'name': 'USDC', 'address': USDC_ADDRESS},
        {'name': 'WETH', 'address': WETH_ADDRESS}
    ]
    
    for pool in pools:
        print(f"\n🔄 Adding {pool['name']} pool...")
        print(f"   Token: {pool['address']}")
        
        # Check if exists
        pool_id = Web3.solidity_keccak(['address', 'address'], [pool['address'], aave_adapter_address])
        
        try:
            pool_data = strategy_manager.functions.getPool(pool_id).call()
            if pool_data[0] != "0x0000000000000000000000000000000000000000":
                print(f"   ✅ {pool['name']} pool already exists")
                continue
        except:
            pass
        
        # Add pool
        try:
            gas_price = w3.eth.gas_price
            gas_price_buffered = int(gas_price * 1.2)
            nonce = w3.eth.get_transaction_count(account.address, 'pending')
            
            transaction = strategy_manager.functions.addPool(
                pool['address'],
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
            
            print(f"   📤 Transaction: {tx_hash.hex()}")
            
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
            
            if receipt['status'] == 1:
                print(f"   ✅ {pool['name']} pool added! Gas used: {receipt['gasUsed']}")
            else:
                print(f"   ❌ Transaction failed")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ POOLS SETUP COMPLETE!")
    print("=" * 60)
    print("\n📋 Next: Check your token balances")
    print("   python scripts/check_balances_base.py")

if __name__ == '__main__':
    add_pools_base()
