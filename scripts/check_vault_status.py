import sys
sys.path.append('.')
from src.execution.contract_manager import ContractManager

cm = ContractManager('base_sepolia')
vault = cm.contracts['YieldVault']
w3 = cm.w3
user = cm.account.address
weth = '0x4200000000000000000000000000000000000006'

print('\n🏦 VAULT STATUS')
print('='*60)

try:
    shares = vault.functions.shares(user).call()
    print(f'Your shares: {w3.from_wei(shares, "ether")} shares')
except Exception as e:
    print(f'Could not read shares: {e}')

try:
    tvl = vault.functions.totalValueLocked().call()
    print(f'Total TVL: {w3.from_wei(tvl, "ether")} ETH')
except Exception as e:
    print(f'Could not read TVL: {e}')

supported = vault.functions.isAssetSupported(weth).call()
print(f'WETH supported: {"✅" if supported else "❌"}')

print('='*60)
