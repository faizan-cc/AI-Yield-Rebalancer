"""
Wrap ETH to WETH on Base Sepolia
"""
import os
import sys
from pathlib import Path
from web3 import Web3
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
load_dotenv()

WETH_ADDRESS = "0x4200000000000000000000000000000000000006"

# WETH ABI
WETH_ABI = [
    {"constant": False, "inputs": [], "name": "deposit", "outputs": [], "payable": True, "stateMutability": "payable", "type": "function"},
    {"constant": True, "inputs": [{"name": "", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "payable": False, "stateMutability": "view", "type": "function"}
]

def wrap_eth_base(amount_eth: float):
    """Wrap ETH to WETH on Base Sepolia"""
    
    rpc_url = os.getenv('BASE_SEPOLIA_RPC_URL')
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    
    print("=" * 60)
    print("WRAP ETH TO WETH - BASE SEPOLIA")
    print("=" * 60)
    print(f"\n✅ Connected (Chain ID: {w3.eth.chain_id})")
    
    private_key = os.getenv('DEPLOYER_PRIVATE_KEY')
    account = w3.eth.account.from_key(private_key)
    
    eth_balance = w3.eth.get_balance(account.address)
    eth_balance_formatted = w3.from_wei(eth_balance, 'ether')
    
    print(f"📍 Wallet: {account.address}")
    print(f"💰 ETH Balance: {eth_balance_formatted} ETH")
    
    if eth_balance_formatted < amount_eth:
        print(f"❌ Insufficient ETH. Need {amount_eth}, have {eth_balance_formatted}")
        return
    
    weth_contract = w3.eth.contract(address=WETH_ADDRESS, abi=WETH_ABI)
    weth_balance_before = weth_contract.functions.balanceOf(account.address).call()
    
    print(f"💎 Current WETH: {w3.from_wei(weth_balance_before, 'ether')} WETH")
    print(f"\n🔄 Wrapping {amount_eth} ETH to WETH...")
    
    try:
        amount_wei = w3.to_wei(amount_eth, 'ether')
        gas_price = w3.eth.gas_price
        gas_price_buffered = int(gas_price * 1.2)
        nonce = w3.eth.get_transaction_count(account.address, 'pending')
        
        transaction = weth_contract.functions.deposit().build_transaction({
            'from': account.address,
            'value': amount_wei,
            'gas': 50000,
            'gasPrice': gas_price_buffered,
            'nonce': nonce,
            'chainId': w3.eth.chain_id
        })
        
        signed_txn = account.sign_transaction(transaction)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        print(f"📤 Transaction: {tx_hash.hex()}")
        
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        
        if receipt['status'] == 1:
            weth_balance_after = weth_contract.functions.balanceOf(account.address).call()
            print(f"✅ Success! Gas used: {receipt['gasUsed']}")
            print(f"💎 New WETH Balance: {w3.from_wei(weth_balance_after, 'ether')} WETH")
            print(f"💰 Remaining ETH: {w3.from_wei(w3.eth.get_balance(account.address), 'ether')} ETH")
        else:
            print(f"❌ Transaction failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Wrap ETH to WETH on Base Sepolia')
    parser.add_argument('--amount', type=float, required=True, help='Amount of ETH to wrap')
    args = parser.parse_args()
    
    if args.amount <= 0:
        print("❌ Amount must be positive")
        sys.exit(1)
    
    wrap_eth_base(args.amount)
