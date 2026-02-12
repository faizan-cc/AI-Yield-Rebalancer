import sys
sys.path.append('.')
from src.execution.contract_manager import ContractManager
from web3 import Web3

# The transactions from the keeper log
tx_hashes = [
    "0xddced6868d37300424390d11682a024d03f3d3546223df98ccb53e778244291d",  # ML update
    "0xe0ea1f209ab1e7c9670a6386025562d00889be2cd6076a4378a95b6364df047a"   # Rebalance
]

cm = ContractManager('base_sepolia')
w3 = cm.w3

print('\n🔍 CHECKING BASE SEPOLIA TRANSACTIONS')
print('='*60)

for tx_hash in tx_hashes:
    print(f'\nTransaction: {tx_hash}')
    try:
        tx = w3.eth.get_transaction(tx_hash)
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        
        print(f'  ✅ Found on Base Sepolia!')
        print(f'  Block: {receipt["blockNumber"]}')
        print(f'  Status: {"✅ Success" if receipt["status"] == 1 else "❌ Failed"}')
        print(f'  From: {tx["from"]}')
        print(f'  To: {tx["to"]}')
        print(f'  Gas Used: {receipt["gasUsed"]:,}')
        print(f'  View: https://base-sepolia.blockscout.com/tx/{tx_hash}')
    except Exception as e:
        print(f'  ❌ Not found: {e}')

print('\n' + '='*60)
print('\n🔍 CHECKING WHAT POOLS ARE ACTUALLY CONFIGURED')
print('='*60)

strategy = cm.contracts['StrategyManager']

# Check WETH pools
weth = '0x4200000000000000000000000000000000000006'
usdc_aave = '0x036CbD53842c5426634e7929541eC2318f3dCF7e'

for name, asset in [("WETH", weth), ("USDC (Aave)", usdc_aave)]:
    print(f'\n{name}: {asset}')
    try:
        pools = strategy.functions.getAssetPools(asset).call()
        print(f'  Pools: {len(pools)}')
        for pool in pools:
            print(f'    - {pool}')
    except Exception as e:
        print(f'  Error: {e}')
