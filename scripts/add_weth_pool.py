"""
Add WETH pool to StrategyManager for multi-asset testing
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

def add_weth_pool():
    """Add WETH pool to StrategyManager"""
    
    print("=" * 60)
    print("ADD WETH POOL TO STRATEGY MANAGER")
    print("=" * 60)
    
    # Initialize contract manager
    contract_manager = ContractManager('sepolia')
    
    strategy_manager = contract_manager.contracts['StrategyManager']
    aave_adapter = contract_manager.contracts['AaveAdapter']
    account = contract_manager.account
    w3 = contract_manager.w3
    
    print(f"📍 Wallet: {account.address}")
    print(f"💎 WETH Address: {WETH_ADDRESS}")
    print(f"🔧 Aave Adapter: {aave_adapter.address}")
    
    # Check if pool already exists
    try:
        pool_info = strategy_manager.functions.getPool(1).call()  # Pool ID 1 might be WETH
        print(f"\n⚠️  Pool 1 exists: {pool_info}")
        
        # Try pool ID 2
        try:
            pool_info2 = strategy_manager.functions.getPool(2).call()
            print(f"⚠️  Pool 2 exists: {pool_info2}")
            print("\n✅ WETH pool may already be registered")
            return
        except:
            print("Pool ID 2 is available")
    except Exception as e:
        print(f"Pool 1 does not exist yet: {e}")
    
    print(f"\n🔄 Adding WETH pool...")
    
    try:
        # Get gas price and nonce
        gas_price = w3.eth.gas_price
        gas_price_buffered = int(gas_price * 1.2)
        nonce = w3.eth.get_transaction_count(account.address, 'pending')
        
        # Build transaction to add pool
        transaction = strategy_manager.functions.addPool(
            WETH_ADDRESS,
            aave_adapter.address,
            "AaveV3"  # Protocol name
        ).build_transaction({
            'from': account.address,
            'gas': 200000,
            'gasPrice': gas_price_buffered,
            'nonce': nonce,
            'chainId': w3.eth.chain_id
        })
        
        # Sign and send
        signed_txn = account.sign_transaction(transaction)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        print(f"📤 Transaction sent: {tx_hash.hex()}")
        print("⏳ Waiting for confirmation...")
        
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        
        if receipt['status'] == 1:
            print(f"✅ WETH pool added successfully!")
            print(f"⛽ Gas used: {receipt['gasUsed']}")
            
            # Get the new pool info
            try:
                # Find the pool ID by checking recent pools
                for pool_id in range(0, 10):
                    try:
                        pool = strategy_manager.functions.getPool(pool_id).call()
                        if pool[0].lower() == WETH_ADDRESS.lower():
                            print(f"\n📊 WETH Pool Info (ID: {pool_id}):")
                            print(f"   Asset: {pool[0]}")
                            print(f"   Adapter: {pool[1]}")
                            print(f"   Total Deposited: {pool[2] / 10**18} WETH")
                            print(f"   Expected APY: {pool[3] / 100}%")
                            break
                    except:
                        continue
            except Exception as e:
                print(f"Could not fetch pool info: {e}")
        else:
            print(f"❌ Transaction failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    add_weth_pool()
