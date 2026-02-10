"""
Find Uniswap V3 pool IDs from DefiLlama
"""
import requests
import sys
sys.path.append('src')

from ingestion.backfill_client import DefiLlamaBackfiller

backfiller = DefiLlamaBackfiller()

print("Searching for Uniswap V3 pools on DefiLlama...\n")

uniswap_pools = [
    ('USDC/WETH', '0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640'),
    ('WBTC/WETH', '0xCBCdF9626bC03E24f779434178A73a0B4bad62eD'),
    ('USDC/USDT', '0x3416cF6C708Da44DB2624D63ea0AAef7113527C6'),
    ('DAI/USDC', '0x5777d92f208679DB4b9778590Fa3CAB3aC9e2168'),
]

# Search by pool address directly
url = "https://yields.llama.fi/pools"
print("Fetching all pools from DefiLlama...")
response = requests.get(url, timeout=30)
data = response.json()

print(f"Total pools available: {len(data['data'])}\n")

results = {}
for symbol, address in uniswap_pools:
    print(f"Searching for {symbol}...")
    
    # Extract token symbols from pair
    tokens = symbol.split('/')
    
    # Find by symbol and project
    matches = [
        p for p in data['data']
        if 'uniswap' in p.get('project', '').lower()
        and p.get('chain', '').lower() == 'ethereum'
        and all(token in p.get('symbol', '') for token in tokens)
    ]
    
    # Sort by TVL to get the main pool
    matches.sort(key=lambda x: x.get('tvlUsd', 0), reverse=True)
    
    if matches:
        # Show top 3 matches
        print(f"  Found {len(matches)} matches, showing top 3:")
        for i, pool in enumerate(matches[:3], 1):
            pool_id = pool['pool']
            symbol_str = pool.get('symbol', '')
            tvl = pool.get('tvlUsd', 0)
            apy = pool.get('apy', 0)
            print(f"    {i}. {symbol_str} | TVL: ${tvl:,.0f} | APY: {apy:.2f}%")
            print(f"       Pool: {pool_id}")
            print(f"       Address: {pool.get('pool', 'N/A')}")
            
            # Save the top match
            if i == 1:
                results[symbol] = pool_id
    else:
        print(f"  ✗ Not found")
    print()

print("\n" + "="*60)
print("Pool Mappings for live_collector.py:")
print("="*60)
for symbol, pool_id in results.items():
    print(f"('{symbol}', 'uniswap_v3'): '{pool_id}',")
