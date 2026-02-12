"""
Add DAI pool to StrategyManager
DAI is actually supported by Aave V3 on Sepolia with 71% APY!
"""
import os
import sys
from pathlib import Path
from web3 import Web3
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from src.execution.contract_manager import ContractManager

load_dotenv()

DAI_ADDRESS = "0xFF34B3d4Aee8ddCd6F9AFFFB6Fe49bD371b8a357"

def add_dai_pool():
    """Add DAI pool to StrategyManager"""
    
    print("=" * 60)
    print("ADD DAI POOL TO STRATEGY MANAGER")
    print("=" * 60)
    
    contract_manager = ContractManager('sepolia')
    strategy_manager = contract_manager.contracts['StrategyManager']
    aave_adapter = contract_manager.contracts['AaveAdapter']
    account = contract_manager.account
    w3 = contract_manager.w3
    
    print(f"\n📍 Wallet: {account.address}")
    print(f"💎 DAI Address: {DAI_ADDRESS}")
    print(f"🔧 Aave Adapter: {aave_adapter.address}")
    print(f"✅ DAI is supported on Aave V3 Sepolia (71.10% APY)")
    
    # Check if pool exists
    pool_id = Web3.solidity_keccak(['address', 'address'], [DAI_ADDRESS, aave_adapter.address])
    print(f"\n🔍 Pool ID: {pool_id.hex()}")
    
    try:
        pool_data = strategy_manager.functions.getPool(pool_id).call()
        if pool_data[0] != "0x0000000000000000000000000000000000000000":
            print(f"✅ DAI pool already exists!")
            print(f"   Token: {pool_data[0]}")
            print(f"   Protocol: {pool_data[1]}")
            print(f"   Name: {pool_data[2]}")
            return
    except:
        pass
    
    print(f"\n🔄 Adding DAI pool...")
    
    try:
        # Get gas price and nonce
        gas_price = w3.eth.gas_price
        gas_price_buffered = int(gas_price * 1.2)
        nonce = w3.eth.get_transaction_count(account.address, 'pending')
        
        # Build transaction
        transaction = strategy_manager.functions.addPool(
            DAI_ADDRESS,
            aave_adapter.address,
            "AaveV3"
        ).build_transaction({
            'from': account.address,
            'gas': 250000,  # Increased gas limit
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
            print(f"✅ DAI pool added successfully!")
            print(f"⛽ Gas used: {receipt['gasUsed']}")
            
            # Verify pool was added
            pool_data = strategy_manager.functions.getPool(pool_id).call()
            print(f"\n📊 DAI Pool Info:")
            print(f"   Pool ID: {pool_id.hex()}")
            print(f"   Asset: {pool_data[0]}")
            print(f"   Protocol: {pool_data[1]}")
            print(f"   Name: {pool_data[2]}")
            print(f"   Is Active: {pool_data[6]}")
            
        else:
            print(f"❌ Transaction failed")
            print(f"Receipt: {receipt}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    add_dai_pool()
