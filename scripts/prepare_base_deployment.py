"""
Deploy contracts to Base Sepolia
"""
import os
import sys
import json
from pathlib import Path
from web3 import Web3
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()

# Base Sepolia Aave V3 addresses
BASE_SEPOLIA_AAVE_POOL = "0x07eA79F68B2B3df564D0A34F8e19D9B1e339814b"  # Aave V3 Pool on Base Sepolia

def deploy_to_base():
    """Deploy all contracts to Base Sepolia"""
    
    print("=" * 60)
    print("DEPLOYING TO BASE SEPOLIA")
    print("=" * 60)
    
    # Setup Web3
    rpc_url = os.getenv('BASE_SEPOLIA_RPC_URL')
    if not rpc_url:
        print("❌ BASE_SEPOLIA_RPC_URL not found in .env")
        return
    
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print("❌ Failed to connect to Base Sepolia")
        return
    
    print(f"✅ Connected to Base Sepolia")
    print(f"   Chain ID: {w3.eth.chain_id}")
    print(f"   Block number: {w3.eth.block_number}")
    
    # Get account
    private_key = os.getenv('DEPLOYER_PRIVATE_KEY')
    if not private_key:
        print("❌ DEPLOYER_PRIVATE_KEY not found in .env")
        return
    
    account = w3.eth.account.from_key(private_key)
    balance = w3.eth.get_balance(account.address)
    
    print(f"\n📍 Deployer: {account.address}")
    print(f"💰 Balance: {w3.from_wei(balance, 'ether')} ETH")
    
    if balance < w3.to_wei(0.01, 'ether'):
        print("⚠️  Low balance! You may need more ETH for deployment")
        print("   Get Base Sepolia ETH from: https://www.alchemy.com/faucets/base-sepolia")
    
    print(f"\n🏦 Aave V3 Pool: {BASE_SEPOLIA_AAVE_POOL}")
    print("\n" + "=" * 60)
    print("Ready to deploy contracts")
    print("=" * 60)
    print("\nContracts to deploy:")
    print("  1. AaveAdapter")
    print("  2. UniswapAdapter")
    print("  3. StrategyManager")
    print("  4. RebalanceExecutor")
    print("  5. YieldVault")
    print("\nRun: npx hardhat run scripts/deploy.js --network base_sepolia")

if __name__ == '__main__':
    deploy_to_base()
