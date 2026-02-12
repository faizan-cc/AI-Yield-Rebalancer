"""
Check all token balances and Aave support on Base Sepolia
"""
import os
import sys
from pathlib import Path
from web3 import Web3
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()

# All possible tokens on Base Sepolia
TOKENS = {
    'USDC (yours)': '0xba50Cd2A20f6DA35D788639E581bca8d0B5d4D5f',
    'USDC (Aave)': '0x036CbD53842c5426634e7929541eC2318f3dCF7e',
    'WETH': '0x4200000000000000000000000000000000000006',
    'USDbC': '0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA',
}

AAVE_POOL = "0x07eA79F68B2B3df564D0A34F8e19D9B1e339814b"

def check_all_tokens():
    """Check all tokens - balances and Aave support"""
    
    rpc_url = os.getenv('BASE_SEPOLIA_RPC_URL')
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    
    private_key = os.getenv('DEPLOYER_PRIVATE_KEY')
    account = w3.eth.account.from_key(private_key)
    
    print("=" * 80)
    print("BASE SEPOLIA - TOKEN ANALYSIS")
    print("=" * 80)
    print(f"\n📍 Wallet: {account.address}")
    print(f"⛓️  Chain: Base Sepolia (84532)\n")
    
    # ERC20 ABI
    erc20_abi = [
        {"constant": True, "inputs": [{"name": "_owner", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}], "type": "function"},
        {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
        {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "type": "function"}
    ]
    
    # Aave Pool ABI
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
    
    print(f"{'TOKEN':<15} {'BALANCE':<20} {'AAVE SUPPORT':<15} {'APY'}")
    print("-" * 80)
    
    usable_tokens = []
    
    for name, address in TOKENS.items():
        try:
            # Get balance
            contract = w3.eth.contract(address=address, abi=erc20_abi)
            balance = contract.functions.balanceOf(account.address).call()
            
            try:
                decimals = contract.functions.decimals().call()
                symbol = contract.functions.symbol().call()
            except:
                decimals = 18
                symbol = name
            
            balance_formatted = balance / (10 ** decimals)
            
            # Check Aave support
            try:
                reserve_data = aave_pool.functions.getReserveData(address).call()
                atoken_address = reserve_data[8]
                liquidity_rate = reserve_data[2]
                
                if atoken_address != "0x0000000000000000000000000000000000000000":
                    apy = (liquidity_rate / 1e25) if liquidity_rate > 0 else 0
                    aave_status = "✅ ACTIVE"
                    
                    if balance > 0:
                        usable_tokens.append({
                            'name': name,
                            'symbol': symbol,
                            'address': address,
                            'balance': balance_formatted,
                            'apy': apy,
                            'atoken': atoken_address
                        })
                else:
                    aave_status = "❌ NO"
                    apy = 0
            except:
                aave_status = "❌ ERROR"
                apy = 0
            
            balance_str = f"{balance_formatted:.2f} {symbol}"
            apy_str = f"{apy:.2f}%" if apy > 0 else "N/A"
            
            print(f"{name:<15} {balance_str:<20} {aave_status:<15} {apy_str}")
            
        except Exception as e:
            print(f"{name:<15} {'ERROR':<20} {str(e)[:15]:<15}")
    
    print("\n" + "=" * 80)
    
    if usable_tokens:
        print(f"\n✅ USABLE TOKENS (Have balance + Aave support): {len(usable_tokens)}")
        for token in usable_tokens:
            print(f"\n   • {token['name']} ({token['symbol']})")
            print(f"     Address: {token['address']}")
            print(f"     Balance: {token['balance']:.2f} {token['symbol']}")
            print(f"     APY: {token['apy']:.4f}%")
            print(f"     aToken: {token['atoken']}")
    else:
        print("\n❌ NO USABLE TOKENS FOUND")
        print("\nℹ️  You have tokens but they're not supported by Aave on Base Sepolia")
        print("   Recommendation: Get WETH by wrapping ETH")
        print("   Command: python scripts/wrap_eth.py --amount 0.01")

if __name__ == '__main__':
    check_all_tokens()
