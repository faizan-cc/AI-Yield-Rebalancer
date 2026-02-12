import psycopg2

conn = psycopg2.connect(
    dbname='defi_yield_db',
    user='postgres',
    password='postgres',
    host='localhost'
)
cur = conn.cursor()

# Get transaction count
cur.execute("""
    SELECT COUNT(DISTINCT update_tx_hash) 
    FROM ml_predictions 
    WHERE network='base_sepolia' 
    AND update_tx_hash IS NOT NULL
""")
tx_count = cur.fetchone()[0]

# Get APY history
cur.execute("""
    SELECT predicted_apy, timestamp 
    FROM ml_predictions 
    WHERE network='base_sepolia' 
    ORDER BY timestamp
""")
data = cur.fetchall()
apys = [r[0] for r in data]

print('\n📈 APY TREND ANALYSIS')
print('='*60)
print(f'Total Predictions: {len(apys)}')
print(f'On-chain Transactions: {tx_count}')
print(f'APY Range: {min(apys):.2f}% - {max(apys):.2f}%')
print(f'Starting APY: {apys[0]:.2f}% ({data[0][1]})')
print(f'Current APY: {apys[-1]:.2f}% ({data[-1][1]})')
print(f'Change: {((apys[-1] - apys[0])/apys[0]*100):+.1f}%')

print(f'\nRecent Trend (last 10 predictions):')
for i, (apy, ts) in enumerate(data[-10:], 1):
    print(f'  {i:2d}. {apy:5.2f}% at {ts.strftime("%H:%M:%S")}')

# Gas usage stats
cur.execute("""
    SELECT COUNT(*), SUM(CAST(SUBSTRING(update_tx_hash FROM 3) AS bigint)) 
    FROM ml_predictions 
    WHERE network='base_sepolia' 
    AND update_tx_hash IS NOT NULL 
    AND update_tx_hash != ''
""")

print('\n' + '='*60)

cur.close()
conn.close()
