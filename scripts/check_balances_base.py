"""
Check token balances on Base Sepolia
"""
import os
import sys
from pathlib import Path
from web3 import Web3
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()

# Base Sepolia tokens
TOKENS = {
    'USDC': '0x036CbD53842c5426634e7929541eC2318f3dCF7e',
    'WETH': '0x4200000000000000000000000000000000000006',
}

# ERC20 ABI
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    }
]

def check_balances():
    """Check balances on Base Sepolia"""
    
    rpc_url = os.getenv('BASE_SEPOLIA_RPC_URL')
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    
    private_key = os.getenv('DEPLOYER_PRIVATE_KEY')
    account = w3.eth.account.from_key(private_key)
    
    print("=" * 60)
    print("BASE SEPOLIA TOKEN BALANCES")
    print("=" * 60)
    print(f"\n📍 Wallet: {account.address}")
    print(f"⛓️  Chain ID: {w3.eth.chain_id}")
    print()
    
    # Check ETH
    eth_balance = w3.eth.get_balance(account.address)
    print(f"{'ETH':<8} {w3.from_wei(eth_balance, 'ether'):>15.6f} ETH")
    
    # Check tokens
    for symbol, address in TOKENS.items():
        try:
            contract = w3.eth.contract(address=address, abi=ERC20_ABI)
            balance = contract.functions.balanceOf(account.address).call()
            
            try:
                decimals = contract.functions.decimals().call()
            except:
                decimals = 18
            
            balance_formatted = balance / (10 ** decimals)
            print(f"{symbol:<8} {balance_formatted:>15.6f} {symbol}")
            
        except Exception as e:
            print(f"{symbol:<8} {'ERROR':>15}")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    check_balances()
