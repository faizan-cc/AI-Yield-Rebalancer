"""
Check token balances (ETH, WETH, USDC, DAI) on Sepolia
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

# Token addresses on Sepolia
TOKENS = {
    'WETH': '0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14',
    'USDC': '0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238',
    'DAI': '0xFF34B3d4Aee8ddCd6F9AFFFB6Fe49bD371b8a357'  # Aave Sepolia DAI
}

# ERC20 ABI (balanceOf and decimals)
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
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    }
]

def check_balances():
    """Check all token balances"""
    
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
    
    # Get wallet address
    private_key = os.getenv('DEPLOYER_PRIVATE_KEY') or os.getenv('PRIVATE_KEY')
    if not private_key:
        print("❌ DEPLOYER_PRIVATE_KEY or PRIVATE_KEY not found in .env")
        return
    
    account = w3.eth.account.from_key(private_key)
    wallet_address = account.address
    
    print(f"📍 Wallet: {wallet_address}")
    print()
    print("=" * 60)
    print("TOKEN BALANCES")
    print("=" * 60)
    
    # Check ETH balance
    eth_balance = w3.eth.get_balance(wallet_address)
    eth_balance_formatted = w3.from_wei(eth_balance, 'ether')
    print(f"{'ETH':<6} {eth_balance_formatted:>15.6f} ETH")
    
    # Check ERC20 token balances
    for symbol, address in TOKENS.items():
        try:
            contract = w3.eth.contract(address=address, abi=ERC20_ABI)
            
            # Get balance
            balance = contract.functions.balanceOf(wallet_address).call()
            
            # Get decimals
            try:
                decimals = contract.functions.decimals().call()
            except:
                decimals = 18  # Default to 18 if can't fetch
            
            # Format balance
            balance_formatted = balance / (10 ** decimals)
            
            print(f"{symbol:<6} {balance_formatted:>15.6f} {symbol}")
            
        except Exception as e:
            print(f"{symbol:<6} {'ERROR':>15} (Could not fetch)")
    
    print("=" * 60)
    
    # Check vault deposits if contract exists
    try:
        from src.execution.contract_manager import ContractManager
        
        contract_manager = ContractManager('sepolia')
        vault = contract_manager.get_contract('YieldVault')
        
        print()
        print("VAULT DEPOSITS")
        print("=" * 60)
        
        # Check deposits for each token
        for symbol, address in TOKENS.items():
            try:
                user_info = vault.functions.getUserInfo(wallet_address, address).call()
                deposit_amount = user_info[0] / (10 ** 18)  # Assuming 18 decimals
                
                if deposit_amount > 0:
                    print(f"{symbol:<6} {deposit_amount:>15.6f} {symbol} (deposited in vault)")
            except:
                pass
        
        # Check USDC deposit (already know this works)
        try:
            usdc_info = vault.functions.getUserInfo(wallet_address, TOKENS['USDC']).call()
            usdc_deposit = usdc_info[0] / (10 ** 6)  # USDC has 6 decimals
            if usdc_deposit > 0:
                print(f"{'USDC':<6} {usdc_deposit:>15.6f} USDC (deposited in vault)")
        except:
            pass
        
        print("=" * 60)
        
    except Exception as e:
        # Vault info not critical
        pass

if __name__ == '__main__':
    check_balances()
