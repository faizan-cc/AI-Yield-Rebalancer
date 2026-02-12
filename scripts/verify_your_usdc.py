import sys
sys.path.append('.')
from src.execution.contract_manager import ContractManager
from web3 import Web3

cm = ContractManager('base_sepolia')
w3 = cm.w3

# Your USDC address
your_usdc = "0xba50Cd2A20f6DA35D788639E581bca8d0B5d4D5f"
# The other USDC we found
other_usdc = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"

# Aave V3 Pool address on Base Sepolia
aave_pool = "0x07eA79F68B2B3df564D0A34F8e19D9B1e339814b"

print('\n🔍 CHECKING BOTH USDC ADDRESSES WITH AAVE')
print('='*70)

# Aave Pool ABI for getReserveData
aave_pool_abi = [
    {
        "inputs": [{"internalType": "address", "name": "asset", "type": "address"}],
        "name": "getReserveData",
        "outputs": [
            {"internalType": "uint256", "name": "configuration", "type": "uint256"},
            {"internalType": "uint128", "name": "liquidityIndex", "type": "uint128"},
            {"internalType": "uint128", "name": "variableBorrowIndex", "type": "uint128"},
            {"internalType": "uint128", "name": "currentLiquidityRate", "type": "uint128"},
            {"internalType": "uint128", "name": "currentVariableBorrowRate", "type": "uint128"},
            {"internalType": "uint128", "name": "currentStableBorrowRate", "type": "uint128"},
            {"internalType": "uint40", "name": "lastUpdateTimestamp", "type": "uint40"},
            {"internalType": "address", "name": "aTokenAddress", "type": "address"},
            {"internalType": "address", "name": "stableDebtTokenAddress", "type": "address"},
            {"internalType": "address", "name": "variableDebtTokenAddress", "type": "address"},
            {"internalType": "address", "name": "interestRateStrategyAddress", "type": "address"},
            {"internalType": "uint8", "name": "id", "type": "uint8"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

pool = w3.eth.contract(address=aave_pool, abi=aave_pool_abi)

# ERC20 ABI
erc20_abi = [
    {"constant":True,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"type":"function"},
    {"constant":True,"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"},
    {"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"}
]

for label, usdc_addr in [("YOUR USDC", your_usdc), ("OTHER USDC", other_usdc)]:
    print(f'\n{label}: {usdc_addr}')
    print('-'*70)
    
    try:
        # Check token info
        token = w3.eth.contract(address=usdc_addr, abi=erc20_abi)
        symbol = token.functions.symbol().call()
        decimals = token.functions.decimals().call()
        balance = token.functions.balanceOf(cm.account.address).call()
        
        print(f'  Symbol: {symbol}')
        print(f'  Decimals: {decimals}')
        print(f'  Your Balance: {balance / 10**decimals:,.2f} {symbol}')
        
        # Check Aave reserve data
        reserve_data = pool.functions.getReserveData(usdc_addr).call()
        
        atoken = reserve_data[7]
        liquidity_rate = reserve_data[3]
        
        if atoken == '0x0000000000000000000000000000000000000000':
            print(f'  ❌ NOT supported by Aave (no aToken)')
        else:
            apy = (liquidity_rate / 10**27) * 100
            print(f'  ✅ SUPPORTED by Aave!')
            print(f'  aToken: {atoken}')
            print(f'  Current APY: {apy:.2f}%')
            
    except Exception as e:
        print(f'  ❌ Error: {e}')

print('\n' + '='*70)
print('\n📊 CHECKING YOUR POOLS')
print('='*70)

# Check what pools are registered in StrategyManager
strategy = cm.contracts['StrategyManager']

# Get USDC pool if it exists
try:
    # Try to get pool info by index or asset
    print(f'\nChecking if your USDC is already in StrategyManager...')
    
    # Check pool count
    for i in range(10):
        try:
            pool_addr = strategy.functions.poolAddresses(i).call()
            pool_info = strategy.functions.poolInfo(pool_addr).call()
            asset = pool_info[0]
            apy = pool_info[2] / 100
            
            print(f'\n  Pool {i}: {pool_addr}')
            print(f'    Asset: {asset}')
            print(f'    APY: {apy:.2f}%')
            
            if asset.lower() == your_usdc.lower():
                print(f'    ✅ This is YOUR USDC!')
                
        except:
            break
            
except Exception as e:
    print(f'  Error checking pools: {e}')

print('\n' + '='*70)
