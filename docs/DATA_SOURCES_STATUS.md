# Data Sources Status

## Overview
This document tracks the status of our data integration pipelines for Aave V3, Uniswap V3, and Curve Finance.

## Current Status (Last Updated: 2025-01-18)

### ✅ **Uniswap V3** - FULLY OPERATIONAL
- **Status**: Production Ready
- **Source**: The Graph Decentralized Network
- **Subgraph ID**: `5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV`
- **Endpoint**: `https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV`
- **Data Retrieved**:
  - ✅ Pool liquidity (TVL)
  - ✅ Trading volumes (24h, 7d)
  - ✅ Fee tiers (100, 500, 3000, 10000 bps)
  - ✅ Token pairs
  - ✅ Price data (sqrtPrice, tick)
- **Test Results**: Successfully retrieved 50 pools with TVL data
- **Query Performance**: < 2 seconds for 50 pools

### ⚠️ **Aave V3** - REQUIRES ALTERNATIVE SOURCE
- **Status**: The Graph subgraphs not accessible with current API key
- **Primary Issue**: Official Aave V3 subgraphs not found on decentralized network gateway
- **Alternative Solutions**:
  1. **Dune Analytics** (RECOMMENDED)
     - Query ID: TBD (see `src/data/dune_client.py`)
     - Provides historical APY, TVL, utilization rates
     - Update frequency: ~15 minutes
  2. **Direct RPC Calls** via Alchemy
     - Contract: `0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2` (Aave V3 Pool)
     - Methods: `getReserveData()`, `getUserAccountData()`
  3. **Aave API** (if available)
     - REST API for protocol metrics

### ⚠️ **Curve Finance** - REQUIRES ALTERNATIVE SOURCE  
- **Status**: The Graph subgraphs not accessible with current API key
- **Primary Issue**: Curve subgraphs not found on decentralized network gateway
- **Alternative Solutions**:
  1. **Dune Analytics** (RECOMMENDED)
     - Query ID: TBD (see `src/data/dune_client.py`)
     - Provides pool APYs, TVL, trading volumes
     - Update frequency: ~15 minutes
  2. **Curve Registry Contracts** via RPC
     - Registry: `0x90E00ACe148ca3b23Ac1bC8C240C2a7Dd9c2d7f5`
     - Methods: `get_pool_info()`, `get_virtual_price()`
  3. **Curve API** (if available)
     - REST API for pool metrics

### ✅ **Alchemy RPC** - OPERATIONAL
- **Status**: Production Ready
- **Current Block**: 24,407,945+
- **Gas Price Tracking**: Working (0.05-0.08 gwei)
- **Contract Calls**: Functional
- **Enhanced APIs**: Available (`alchemy_getTokenBalances`)

### 🔄 **Dune Analytics** - CONFIGURED BUT UNTESTED
- **Status**: API key configured, awaiting query creation
- **API Key**: Configured in `.env`
- **Client**: `src/data/dune_client.py` (400+ lines, ready to use)
- **Query Templates**: Pre-built for Aave, Uniswap, Curve
- **Next Step**: Execute test queries

## Recommended Data Strategy

### Phase 1 (Current): Multi-Source Approach
```
Uniswap V3:  The Graph (real-time) ✅
Aave V3:     Dune Analytics (15-min delay) ⚠️ → need to test
Curve:       Dune Analytics (15-min delay) ⚠️ → need to test
Gas Prices:  Alchemy RPC (real-time) ✅
```

### Phase 2 (Future): Optimize Sources
- Investigate The Graph Studio for custom Aave/Curve deployments
- Consider Chainlink oracles for price feeds
- Evaluate Aave/Curve official APIs if available
- Implement caching layer (Redis) to reduce API calls

## The Graph Migration Context

### Why Hosted Service Endpoints Don't Work
- **Sunset Date**: December 2023
- **Migration Required**: All queries must use decentralized network
- **API Key Required**: Gateway authentication mandatory
- **Subgraph Availability**: Not all hosted subgraphs migrated to decentralized network

### Working with Decentralized Network
1. **URL Format**:
   ```
   https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}
   ```
2. **Finding Subgraph IDs**:
   - Explorer: https://thegraph.com/explorer
   - Search by protocol name
   - Check "Deployment ID" on subgraph page
3. **Known Working IDs**:
   - Uniswap V3: `5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV`
   - More to be discovered

## Action Items

### Immediate (Week 2)
- [x] Confirm Uniswap V3 integration working
- [ ] Create Dune Analytics queries for Aave V3
- [ ] Create Dune Analytics queries for Curve Finance
- [ ] Test end-to-end data collection with Uniswap + Dune
- [ ] Document query IDs in this file

### Short-term (Week 3-4)
- [ ] Implement RPC fallback for Aave V3 (direct contract calls)
- [ ] Implement RPC fallback for Curve (registry contracts)
- [ ] Add data quality monitoring (freshness, completeness)
- [ ] Set up Redis caching for API responses

### Long-term (Month 2+)
- [ ] Investigate custom subgraph deployment
- [ ] Evaluate paid data providers (Nansen, DeFiLlama API)
- [ ] Implement multi-source data validation (cross-check sources)
- [ ] Build data reconciliation layer

## Testing Commands

```bash
# Test all data sources
cd /home/faizan/work/Defi-Yield-R\&D
source venv/bin/activate
python scripts/test_data_integration.py

# Test only Uniswap V3
python -c "from src.data.graph_client import GraphClient; import asyncio; asyncio.run(GraphClient().get_uniswap_pools())"

# Test Dune Analytics (once queries created)
python -c "from src.data.dune_client import DuneClient; client = DuneClient(); client.get_aave_protocol_metrics()"
```

## References

- **The Graph Explorer**: https://thegraph.com/explorer
- **Dune Analytics**: https://dune.com/
- **Aave V3 Contract**: https://etherscan.io/address/0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2
- **Curve Registry**: https://etherscan.io/address/0x90E00ACe148ca3b23Ac1bC8C240C2a7Dd9c2d7f5
