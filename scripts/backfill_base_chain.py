"""
Backfill Historical Data for Base Chain

Fetches 1+ year of historical yield data from DefiLlama for Base chain pools.
This is essential for ML model training with Base chain data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.backfill_client import DefiLlamaBackfiller
from datetime import datetime, timedelta


def backfill_base_chain():
    """Backfill all Base chain pools with 1 year of historical data."""
    
    backfiller = DefiLlamaBackfiller()
    
    # Define Base chain pools to backfill
    # Format: (symbol, protocol, chain, pool_id)
    base_pools = [
        # Aave V3 on Base
        ('USDC', 'aave_v3', 'base', '7e0661bf-8cf3-45e6-9424-31916d4c7b84'),
        ('cbBTC', 'aave_v3', 'base', '89bc7c4c-d71c-435c-ab28-56c803d51320'),
        ('WETH', 'aave_v3', 'base', 'f0131970-afac-4835-b22c-520f192e01d5'),
        
        # Morpho on Base
        ('USDC', 'morpho', 'base', '7820bd3c-461a-4811-9f0b-1d39c1503c3f'),
        ('cbBTC', 'morpho', 'base', '7d33d57d-36dc-414b-9538-22a223250468'),
        
        # Uniswap V3 on Base
        ('WETH/USDC', 'uniswap_v3', 'base', 'b99bcdf5-1350-4269-981e-0e9b5cccb007'),
        ('WETH/cbBTC', 'uniswap_v3', 'base', 'ae6e650d-2da1-43ee-b960-2adfdf4dc2b7'),
        ('USDC/cbBTC', 'uniswap_v3', 'base', '9c3c95ef-5e04-4c75-b7ec-6a59a9ea904b'),
        
        # Aerodrome on Base
        ('USDC/AERO', 'aerodrome', 'base', 'd32f9c01-47d1-4077-8c73-8b91b08d1e91'),
        ('WETH/USDC', 'aerodrome', 'base', 'e8cb4dbb-9e66-4cfa-9c77-407118b128a0'),
    ]
    
    # Calculate date range (1 year of historical data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print("=" * 80)
    print("BASE CHAIN HISTORICAL DATA BACKFILL")
    print("=" * 80)
    print(f"\n📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"📊 Total Pools: {len(base_pools)}")
    print(f"⛓️  Chain: Base (L2 - Lower Fees)")
    print("\n" + "=" * 80 + "\n")
    
    success_count = 0
    failed_pools = []
    
    for i, (symbol, protocol, chain, pool_id) in enumerate(base_pools, 1):
        print(f"\n[{i}/{len(base_pools)}] Processing: {protocol} - {symbol} on {chain}")
        print("-" * 60)
        
        try:
            # Get asset ID from database
            asset_id = backfiller.get_asset_id(symbol, protocol, chain)
            
            if not asset_id:
                print(f"  ✗ Asset not found in database: {symbol}/{protocol}/{chain}")
                print(f"    Run: python scripts/add_base_chain_assets.py")
                failed_pools.append((symbol, protocol, "Not in database"))
                continue
            
            print(f"  ✓ Asset ID: {asset_id}")
            
            # Fetch historical data
            print(f"  📥 Fetching historical data from DefiLlama...")
            df = backfiller.get_pool_history(pool_id)
            
            if df is None or df.empty:
                print(f"  ✗ No data returned for {symbol}")
                failed_pools.append((symbol, protocol, "No data"))
                continue
            
            print(f"  ✓ Retrieved {len(df)} data points")
            print(f"    Date range: {df['time'].min()} to {df['time'].max()}")
            print(f"    APY range: {df['apy_percent'].min():.2f}% to {df['apy_percent'].max():.2f}%")
            
            # Insert into database
            print(f"  💾 Inserting into database...")
            backfiller.insert_yield_data(asset_id, df)
            
            print(f"  ✅ Successfully backfilled {symbol}/{protocol}")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {symbol}/{protocol}: {str(e)}")
            failed_pools.append((symbol, protocol, str(e)))
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("BACKFILL SUMMARY")
    print("=" * 80)
    print(f"\n✅ Successful: {success_count}/{len(base_pools)}")
    print(f"❌ Failed: {len(failed_pools)}/{len(base_pools)}")
    
    if failed_pools:
        print("\n⚠️  Failed Pools:")
        for symbol, protocol, reason in failed_pools:
            print(f"   - {protocol}/{symbol}: {reason}")
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("  1. Verify data with: psql -U faizan -d defi_yield_db")
    print("     SELECT COUNT(*), MIN(time), MAX(time) FROM yield_metrics")
    print("     WHERE asset_id IN (SELECT id FROM assets WHERE chain = 'base');")
    print("  2. Start live collection: python src/ingestion/scheduler.py")
    print("  3. Retrain ML models: python src/ml/train_models.py")
    print("  4. Run backtest: python src/strategies/backtest.py")
    print("=" * 80)


if __name__ == "__main__":
    backfill_base_chain()
