"""
Wrap Sepolia ETH to WETH for testing
"""
import os
import sys
from pathlib import Path
from web3 import Web3
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# WETH contract on Sepolia
WETH_ADDRESS = "0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14"  # Official WETH on Sepolia

# WETH ABI (deposit and balanceOf functions)
WETH_ABI = [
    {
        "constant": False,
        "inputs": [],
        "name": "deposit",
        "outputs": [],
        "payable": True,
        "stateMutability": "payable",
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [{"name": "wad", "type": "uint256"}],
        "name": "withdraw",
        "outputs": [],
        "payable": False,
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    }
]

def wrap_eth(amount_eth: float):
    """Wrap ETH to WETH"""
    
    # Setup Web3
    rpc_url = os.getenv('SEPOLIA_RPC_URL')
    if not rpc_url:
        print("❌ SEPOLIA_RPC_URL not found in .env")
        return
    
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print("❌ Failed to connect to Sepolia")
        return
    
    print(f"✅ Connected to Sepolia (Chain ID: {w3.eth.chain_id})")
    
    # Get private key
    private_key = os.getenv('DEPLOYER_PRIVATE_KEY') or os.getenv('PRIVATE_KEY')
    if not private_key:
        print("❌ DEPLOYER_PRIVATE_KEY or PRIVATE_KEY not found in .env")
        return
    
    account = w3.eth.account.from_key(private_key)
    print(f"📍 Wallet: {account.address}")
    
    # Check ETH balance
    eth_balance = w3.eth.get_balance(account.address)
    eth_balance_formatted = w3.from_wei(eth_balance, 'ether')
    print(f"💰 ETH Balance: {eth_balance_formatted} ETH")
    
    if eth_balance_formatted < amount_eth:
        print(f"❌ Insufficient ETH balance. Need {amount_eth} ETH, have {eth_balance_formatted} ETH")
        return
    
    # Get WETH contract
    weth_contract = w3.eth.contract(address=WETH_ADDRESS, abi=WETH_ABI)
    
    # Check current WETH balance
    weth_balance_before = weth_contract.functions.balanceOf(account.address).call()
    print(f"💎 Current WETH Balance: {w3.from_wei(weth_balance_before, 'ether')} WETH")
    
    # Convert amount to wei
    amount_wei = w3.to_wei(amount_eth, 'ether')
    
    print(f"\n🔄 Wrapping {amount_eth} ETH to WETH...")
    
    try:
        # Get gas price
        gas_price = w3.eth.gas_price
        gas_price_buffered = int(gas_price * 1.2)  # 20% buffer
        
        # Get nonce
        nonce = w3.eth.get_transaction_count(account.address, 'pending')
        
        # Build transaction
        transaction = weth_contract.functions.deposit().build_transaction({
            'from': account.address,
            'value': amount_wei,
            'gas': 50000,  # deposit is simple, fixed gas
            'gasPrice': gas_price_buffered,
            'nonce': nonce,
            'chainId': w3.eth.chain_id
        })
        
        # Sign and send transaction
        signed_txn = w3.eth.account.sign_transaction(transaction, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        print(f"📤 Transaction sent: {tx_hash.hex()}")
        print("⏳ Waiting for confirmation...")
        
        # Wait for transaction receipt
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        
        if receipt['status'] == 1:
            print(f"✅ Successfully wrapped {amount_eth} ETH to WETH!")
            print(f"⛽ Gas used: {receipt['gasUsed']}")
            
            # Check new WETH balance
            weth_balance_after = weth_contract.functions.balanceOf(account.address).call()
            print(f"💎 New WETH Balance: {w3.from_wei(weth_balance_after, 'ether')} WETH")
            
            # Check remaining ETH
            eth_balance_after = w3.eth.get_balance(account.address)
            print(f"💰 Remaining ETH Balance: {w3.from_wei(eth_balance_after, 'ether')} ETH")
            
        else:
            print(f"❌ Transaction failed!")
            print(f"Receipt: {receipt}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Wrap ETH to WETH on Sepolia')
    parser.add_argument('--amount', type=float, required=True, help='Amount of ETH to wrap (e.g., 0.1)')
    
    args = parser.parse_args()
    
    if args.amount <= 0:
        print("❌ Amount must be positive")
        sys.exit(1)
    
    print("=" * 60)
    print(f"WRAP ETH TO WETH - SEPOLIA TESTNET")
    print("=" * 60)
    
    wrap_eth(args.amount)
