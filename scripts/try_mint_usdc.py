import sys
sys.path.append('.')
from src.execution.contract_manager import ContractManager
from web3 import Web3
import json

usdc_address = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"
cm = ContractManager('base_sepolia')
w3 = cm.w3

print('\n🔍 Checking USDC Contract Functions')
print('='*60)

# Extended ERC20 ABI with common test token functions
test_token_abi = [
    {"constant":True,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"type":"function"},
    {"constant":True,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"type":"function"},
    {"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},
    {"constant":True,"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"},
    {"constant":False,"inputs":[{"name":"to","type":"address"},{"name":"amount","type":"uint256"}],"name":"mint","outputs":[{"name":"","type":"bool"}],"type":"function"},
    {"constant":False,"inputs":[{"name":"amount","type":"uint256"}],"name":"mint","outputs":[],"type":"function"},
    {"constant":False,"inputs":[],"name":"mint","outputs":[],"type":"function"},
    {"constant":False,"inputs":[],"name":"faucet","outputs":[],"type":"function"},
    {"constant":False,"inputs":[{"name":"amount","type":"uint256"}],"name":"faucet","outputs":[],"type":"function"},
    {"constant":False,"inputs":[{"name":"to","type":"address"}],"name":"faucet","outputs":[],"type":"function"},
    {"constant":True,"inputs":[],"name":"owner","outputs":[{"name":"","type":"address"}],"type":"function"},
    {"constant":True,"inputs":[],"name":"minter","outputs":[{"name":"","type":"address"}],"type":"function"}
]

usdc = w3.eth.contract(address=usdc_address, abi=test_token_abi)

print(f'Contract: {usdc_address}')
print(f'Your wallet: {cm.account.address}')
print()

# Try to call mint functions
print('Attempting different mint/faucet patterns:')
print()

# Pattern 1: mint(address, amount)
try:
    amount = 1000 * 10**6  # 1000 USDC
    gas_price = int(w3.eth.gas_price * 1.2)
    nonce = w3.eth.get_transaction_count(cm.account.address, 'pending')
    
    tx = usdc.functions.mint(cm.account.address, amount).build_transaction({
        'from': cm.account.address,
        'gas': 100000,
        'gasPrice': gas_price,
        'nonce': nonce,
        'chainId': w3.eth.chain_id
    })
    
    signed = cm.account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
    
    print(f'✅ mint(address, amount) transaction sent!')
    print(f'   Amount: 1000 USDC')
    print(f'   TX: {tx_hash.hex()}')
    print(f'   Waiting for confirmation...')
    
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    
    if receipt['status'] == 1:
        print(f'   ✅ SUCCESS! Gas used: {receipt["gasUsed"]:,}')
        print(f'   View: https://base-sepolia.blockscout.com/tx/{tx_hash.hex()}')
        
        # Check new balance
        balance = usdc.functions.balanceOf(cm.account.address).call()
        print(f'   New balance: {balance / 10**6} USDC')
    else:
        print(f'   ❌ Transaction failed')
        
except Exception as e:
    error_msg = str(e)
    if 'execution reverted' in error_msg.lower():
        print(f'❌ mint(address, amount) - Reverted (not public or not allowed)')
    elif 'could not identify' in error_msg.lower() or 'not found' in error_msg.lower():
        print(f'❌ mint(address, amount) - Function not found')
    else:
        print(f'❌ mint(address, amount) - Error: {error_msg[:100]}')

print()
print('='*60)
print('\n💡 Recommended: Use Aave Faucet')
print('   https://faucet.aave.com/')
print('   Select "Base Sepolia" and request USDC')
print()
