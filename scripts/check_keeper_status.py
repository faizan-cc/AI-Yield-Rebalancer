import sys
sys.path.append('.')
from src.execution.contract_manager import ContractManager
import subprocess
from datetime import datetime

print('\n' + '='*70)
print('🤖 KEEPER SERVICE STATUS - BASE SEPOLIA')
print('='*70)

# Check if process is running
try:
    result = subprocess.run(
        ['ps', 'aux'], 
        capture_output=True, 
        text=True
    )
    keeper_lines = [line for line in result.stdout.split('\n') if 'keeper_service.py' in line and 'grep' not in line]
    
    if keeper_lines:
        print('\n✅ Keeper is RUNNING')
        for line in keeper_lines:
            parts = line.split()
            pid = parts[1]
            cpu = parts[2]
            mem = parts[3]
            print(f'   PID: {pid}')
            print(f'   CPU: {cpu}%')
            print(f'   Memory: {mem}%')
    else:
        print('\n❌ Keeper is NOT RUNNING')
except Exception as e:
    print(f'\n⚠️  Could not check process: {e}')

# Check Base Sepolia connection
print('\n' + '-'*70)
print('📡 BASE SEPOLIA CONNECTION')
print('-'*70)

try:
    cm = ContractManager('base_sepolia')
    w3 = cm.w3
    
    print(f'✅ Connected to Base Sepolia')
    print(f'   Chain ID: {w3.eth.chain_id}')
    print(f'   Latest Block: {w3.eth.block_number:,}')
    print(f'   Account: {cm.account.address}')
    print(f'   Balance: {w3.from_wei(w3.eth.get_balance(cm.account.address), "ether"):.4f} ETH')
    
    # Check vault status
    vault = cm.contracts['YieldVault']
    weth = '0x4200000000000000000000000000000000000006'
    
    print('\n' + '-'*70)
    print('🏦 VAULT STATUS')
    print('-'*70)
    
    shares = vault.functions.shares(cm.account.address).call()
    tvl = vault.functions.totalValueLocked().call()
    weth_supported = vault.functions.isAssetSupported(weth).call()
    
    print(f'Your Shares: {w3.from_wei(shares, "ether"):.4f}')
    print(f'Total TVL: {w3.from_wei(tvl, "ether"):.4f} ETH')
    print(f'WETH Supported: {"✅ Yes" if weth_supported else "❌ No"}')
    
    # Check recent predictions from database
    print('\n' + '-'*70)
    print('🧠 RECENT ML PREDICTIONS')
    print('-'*70)
    
    import psycopg2
    conn = psycopg2.connect(
        dbname="defi_yield_db",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    
    cur.execute("""
        SELECT timestamp, pool_address, predicted_apy, risk_level, confidence_score, network
        FROM ml_predictions
        WHERE network = 'base_sepolia'
        ORDER BY timestamp DESC
        LIMIT 5
    """)
    
    predictions = cur.fetchall()
    
    if predictions:
        print(f'\nLast {len(predictions)} predictions:')
        for i, pred in enumerate(predictions, 1):
            timestamp, pool, apy, risk, conf, network = pred
            print(f'\n  {i}. {timestamp.strftime("%Y-%m-%d %H:%M:%S")}')
            print(f'     Pool: {pool[:10]}...')
            print(f'     Predicted APY: {apy:.2f}%')
            print(f'     Risk: {risk} (confidence: {conf:.1f}%)')
    else:
        print('\n⚠️  No predictions found for base_sepolia')
    
    cur.close()
    conn.close()
    
except Exception as e:
    print(f'❌ Error: {e}')

print('\n' + '='*70)
print('💡 Commands:')
print('   View logs: tail -f logs/keeper.log')
print('   Stop keeper: pkill -f keeper_service')
print('   Check transactions: https://base-sepolia.blockscout.com/address/0xeFdAAaBAC2d15EcfD192f12e3b4690d4f81bef2B')
print('='*70 + '\n')
