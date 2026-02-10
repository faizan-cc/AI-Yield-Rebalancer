"""
Data Collection Scheduler

Runs backfill once at startup, then collects live data every 15 minutes.
"""

import schedule
import time
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.backfill_client import DefiLlamaBackfiller
from ingestion.live_collector import LiveCollector


def run_backfill():
    """Run historical data backfill (once at startup)."""
    print("\n" + "="*60)
    print("INITIAL BACKFILL - Running once at startup")
    print("="*60)
    
    backfiller = DefiLlamaBackfiller()
    backfiller.backfill_all()


def run_live_collection():
    """Run live data collection."""
    print("\n" + "="*60)
    print(f"LIVE COLLECTION - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("="*60)
    
    collector = LiveCollector()
    collector.collect_and_store()


def main():
    """Main scheduler loop."""
    print("\n" + "="*70)
    print("DeFi Yield Data Collection Scheduler")
    print("="*70)
    print("\n🔄 Schedule:")
    print("  - Backfill: Once at startup")
    print("  - Live collection: Every 1 minute")
    print("\n▶ Starting...\n")
    
    # Run backfill once at startup
    # Uncomment this if you want to backfill on startup
    # run_backfill()
    
    # Run first live collection immediately
    run_live_collection()
    
    # Schedule live collection every 1 minutes
    schedule.every(1).minutes.do(run_live_collection)
    
    print("\n✓ Scheduler running. Press Ctrl+C to stop.\n")
    
    # Run scheduled tasks
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\n\n⏹ Scheduler stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error in scheduler: {str(e)}")
            print("Continuing...")
            time.sleep(60)


if __name__ == "__main__":
    main()
