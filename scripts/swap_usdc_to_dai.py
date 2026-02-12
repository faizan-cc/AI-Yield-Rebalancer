"""
Swap USDC to DAI on Uniswap Sepolia
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

# Uniswap V2 Router on Sepolia
UNISWAP_ROUTER = "0xC532a74256D3Db42D0Bf7a0400fEFDbad7694008"  # Uniswap V2 Router

# Token addresses
USDC_ADDRESS = "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238"
DAI_ADDRESS = "0xFF34B3d4Aee8ddCd6F9AFFFB6Fe49bD371b8a357"  # Aave DAI
WETH_ADDRESS = "0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14"

# Simplified Router ABI
ROUTER_ABI = [
    {
        "inputs": [
            {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
            {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
            {"internalType": "address[]", "name": "path", "type": "address[]"},
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "deadline", "type": "uint256"}
        ],
        "name": "swapExactTokensForTokens",
        "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
            {"internalType": "address[]", "name": "path", "type": "address[]"}
        ],
        "name": "getAmountsOut",
        "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
        "stateMutability": "view",
        "type": "function"
    }
]

# ERC20 ABI
ERC20_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
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

def swap_usdc_to_dai(amount_usdc: float):
    """Swap USDC to DAI on Uniswap"""
    
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
    
    # Get account
    private_key = os.getenv('DEPLOYER_PRIVATE_KEY') or os.getenv('PRIVATE_KEY')
    if not private_key:
        print("❌ DEPLOYER_PRIVATE_KEY not found in .env")
        return
    
    account = w3.eth.account.from_key(private_key)
    print(f"📍 Wallet: {account.address}")
    
    # Get contracts
    usdc_contract = w3.eth.contract(address=USDC_ADDRESS, abi=ERC20_ABI)
    router_contract = w3.eth.contract(address=UNISWAP_ROUTER, abi=ROUTER_ABI)
    
    # Check USDC balance
    usdc_balance = usdc_contract.functions.balanceOf(account.address).call()
    usdc_balance_formatted = usdc_balance / (10 ** 6)  # USDC has 6 decimals
    print(f"💰 USDC Balance: {usdc_balance_formatted} USDC")
    
    if usdc_balance_formatted < amount_usdc:
        print(f"❌ Insufficient USDC. Need {amount_usdc}, have {usdc_balance_formatted}")
        return
    
    # Convert amount to proper decimals
    amount_in = int(amount_usdc * (10 ** 6))  # USDC has 6 decimals
    
    print(f"\n🔄 Swapping {amount_usdc} USDC for DAI...")
    
    try:
        # Step 1: Approve USDC spending
        print("📝 Step 1: Approving USDC...")
        
        gas_price = w3.eth.gas_price
        gas_price_buffered = int(gas_price * 1.2)
        nonce = w3.eth.get_transaction_count(account.address, 'pending')
        
        approve_tx = usdc_contract.functions.approve(
            UNISWAP_ROUTER,
            amount_in
        ).build_transaction({
            'from': account.address,
            'gas': 100000,
            'gasPrice': gas_price_buffered,
            'nonce': nonce,
            'chainId': w3.eth.chain_id
        })
        
        signed_approve = w3.eth.account.sign_transaction(approve_tx, private_key)
        approve_hash = w3.eth.send_raw_transaction(signed_approve.rawTransaction)
        
        print(f"   📤 Approval tx: {approve_hash.hex()}")
        receipt = w3.eth.wait_for_transaction_receipt(approve_hash, timeout=180)
        
        if receipt['status'] != 1:
            print("❌ Approval failed")
            return
        
        print("   ✅ Approval successful")
        
        # Step 2: Get expected output amount
        print("💱 Step 2: Calculating swap rate...")
        
        # Path: USDC -> WETH -> DAI (most likely to have liquidity)
        path = [USDC_ADDRESS, WETH_ADDRESS, DAI_ADDRESS]
        
        try:
            amounts_out = router_contract.functions.getAmountsOut(amount_in, path).call()
            expected_dai = amounts_out[-1] / (10 ** 18)  # DAI has 18 decimals
            min_dai = int(amounts_out[-1] * 0.95)  # 5% slippage tolerance
            
            print(f"   Expected DAI output: ~{expected_dai:.2f} DAI")
        except Exception as e:
            print(f"   ⚠️  Could not calculate exact amount: {e}")
            # Set minimum to 0 (accept any amount)
            min_dai = 0
        
        # Step 3: Execute swap
        print("🔀 Step 3: Executing swap...")
        
        deadline = w3.eth.get_block('latest')['timestamp'] + 600  # 10 minutes
        nonce = w3.eth.get_transaction_count(account.address, 'pending')
        
        swap_tx = router_contract.functions.swapExactTokensForTokens(
            amount_in,
            min_dai,
            path,
            account.address,
            deadline
        ).build_transaction({
            'from': account.address,
            'gas': 300000,
            'gasPrice': gas_price_buffered,
            'nonce': nonce,
            'chainId': w3.eth.chain_id
        })
        
        signed_swap = w3.eth.account.sign_transaction(swap_tx, private_key)
        swap_hash = w3.eth.send_raw_transaction(signed_swap.rawTransaction)
        
        print(f"   📤 Swap tx: {swap_hash.hex()}")
        print("   ⏳ Waiting for confirmation...")
        
        receipt = w3.eth.wait_for_transaction_receipt(swap_hash, timeout=180)
        
        if receipt['status'] == 1:
            print(f"   ✅ Swap successful!")
            print(f"   ⛽ Gas used: {receipt['gasUsed']}")
            
            # Check new DAI balance
            dai_contract = w3.eth.contract(address=DAI_ADDRESS, abi=ERC20_ABI)
            dai_balance = dai_contract.functions.balanceOf(account.address).call()
            dai_balance_formatted = dai_balance / (10 ** 18)
            
            print(f"   💎 DAI balance: {dai_balance_formatted} DAI")
        else:
            print("   ❌ Swap failed")
            
    except Exception as e:
        error_msg = str(e)
        if "insufficient liquidity" in error_msg.lower():
            print(f"❌ Insufficient liquidity in USDC/DAI pool on Sepolia")
            print("   Try getting DAI from a faucet or using USDC+WETH only")
        else:
            print(f"❌ Error: {error_msg}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Swap USDC to DAI on Uniswap Sepolia')
    parser.add_argument('--amount', type=float, required=True, help='Amount of USDC to swap')
    
    args = parser.parse_args()
    
    if args.amount <= 0:
        print("❌ Amount must be positive")
        sys.exit(1)
    
    print("=" * 60)
    print("SWAP USDC TO DAI - UNISWAP SEPOLIA")
    print("=" * 60)
    
    swap_usdc_to_dai(args.amount)
