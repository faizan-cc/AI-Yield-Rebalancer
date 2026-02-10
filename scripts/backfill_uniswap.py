"""
Backfill Uniswap pools using known pool IDs
"""
import sys
sys.path.append('src')

from ingestion.backfill_client import DefiLlamaBackfiller

backfiller = DefiLlamaBackfiller()

# Known Uniswap pool IDs from DefiLlama
uniswap_pools = [
    ('USDC/WETH', 'uniswap_v3', '665dc8bc-c79d-4800-97f7-304bf368e547'),
    ('WBTC/WETH', 'uniswap_v3', 'c5599b3a-ea73-4017-a867-72eb971301d1'),
    ('USDC/USDT', 'uniswap_v3', 'e737d721-f45c-40f0-9793-9f56261862b9'),
    ('DAI/USDC', 'uniswap_v3', 'a86ee795-54d9-4812-9148-b312967cefe5'),
]

print('='*60)
print('Backfilling Uniswap V3 Pools (Direct Pool IDs)')
print('='*60)

for i, (symbol, protocol, pool_id) in enumerate(uniswap_pools, 1):
    print(f'\n[{i}/{len(uniswap_pools)}]')
    print(f'\n{"="*60}')
    print(f'Backfilling {protocol} - {symbol}')
    print(f'{"="*60}')
    
    # Get asset ID
    asset_id = backfiller.get_asset_id(symbol, protocol)
    if not asset_id:
        print(f'  ✗ Asset {symbol} not found in database for {protocol}')
        continue
    
    print(f'  Asset ID: {asset_id}')
    print(f'  Pool ID: {pool_id}')
    
    # Fetch historical data directly with pool ID
    df = backfiller.get_pool_history(pool_id)
    if df.empty:
        continue
    
    # Insert into database
    backfiller.insert_yield_data(asset_id, df)
    
    import time
    time.sleep(1)

print('\n' + '='*60)
print('Uniswap Backfill Complete!')
print('='*60)
backfiller.print_summary()
