"""
Add Base Chain Assets to Database

Adds Base chain pools and assets to the database for data collection.
"""

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# Base chain assets to add
BASE_ASSETS = [
    # (symbol, protocol, chain, address, decimals)
    ('USDC', 'aave_v3', 'base', '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913', 6),
    ('cbBTC', 'aave_v3', 'base', '0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf', 8),
    ('WETH', 'aave_v3', 'base', '0x4200000000000000000000000000000000000006', 18),
    
    ('USDC', 'morpho', 'base', '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913', 6),
    ('cbBTC', 'morpho', 'base', '0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf', 8),
    
    ('WETH/USDC', 'uniswap_v3', 'base', '0xd0b53D9277642d899DF5C87A3966A349A798F224', 18),
    ('WETH/cbBTC', 'uniswap_v3', 'base', '0x1234567890000000000000000000000000000001', 18),  # Unique placeholder
    ('USDC/cbBTC', 'uniswap_v3', 'base', '0x1234567890000000000000000000000000000002', 18),  # Unique placeholder
    
    ('USDC/AERO', 'aerodrome', 'base', '0x1234567890000000000000000000000000000003', 18),  # Unique placeholder
    ('WETH/USDC', 'aerodrome', 'base', '0x1234567890000000000000000000000000000004', 18),  # Unique placeholder
]


def add_base_assets():
    """Add Base chain assets to database."""
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'defi_yield_db'),
        'user': os.getenv('DB_USER', 'faizan'),
        'password': os.getenv('DB_PASSWORD', '')
    }
    
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    
    print("Adding Base Chain assets to database...\n")
    
    added = 0
    skipped = 0
    
    for symbol, protocol, chain, address, decimals in BASE_ASSETS:
        # Check if already exists
        cur.execute(
            "SELECT id FROM assets WHERE symbol = %s AND protocol = %s AND chain = %s",
            (symbol, protocol, chain)
        )
        
        if cur.fetchone():
            print(f"  ⏭  {symbol:15s} | {protocol:12s} | {chain:8s} - Already exists")
            skipped += 1
            continue
        
        # Insert new asset
        cur.execute(
            """INSERT INTO assets (symbol, protocol, chain, address, decimals)
               VALUES (%s, %s, %s, %s, %s)""",
            (symbol, protocol, chain, address, decimals)
        )
        
        print(f"  ✓ {symbol:15s} | {protocol:12s} | {chain:8s} - Added")
        added += 1
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"\n✓ Added {added} new assets, skipped {skipped} existing")
    print(f"\nTotal Base chain assets: {added + skipped}")


if __name__ == "__main__":
    add_base_assets()
