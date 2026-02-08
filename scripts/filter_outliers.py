"""
Filter extreme yield outliers and retrain
Removes pools with >200% APY that skew the model
"""

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv('DB_NAME', 'defi_yield_db'),
    user=os.getenv('DB_USER', 'faizan'),
    password=os.getenv('DB_PASSWORD', ''),
    host=os.getenv('DB_HOST', 'localhost'),
    port=os.getenv('DB_PORT', '5432')
)

cursor = conn.cursor()

# Check distribution
cursor.execute("""
    SELECT 
        CASE 
            WHEN apy_percent > 200 THEN '200%+'
            WHEN apy_percent > 50 THEN '50-200%'
            WHEN apy_percent > 10 THEN '10-50%'
            WHEN apy_percent > 1 THEN '1-10%'
            ELSE '0-1%'
        END as bucket,
        COUNT(*) 
    FROM protocol_yields 
    GROUP BY bucket
    ORDER BY bucket
""")

print("APY Distribution:")
for apy, count in cursor.fetchall():
    print(f"  {apy}: {count} records")

# Delete extreme outliers
cursor.execute("DELETE FROM protocol_yields WHERE apy_percent > 200")
conn.commit()
removed = cursor.rowcount
print(f"\n✅ Removed {removed} extreme outlier records (>200% APY)")

# Check new count
cursor.execute("SELECT COUNT(*) FROM protocol_yields")
total = cursor.fetchone()[0]
print(f"New total: {total} records")

conn.close()
