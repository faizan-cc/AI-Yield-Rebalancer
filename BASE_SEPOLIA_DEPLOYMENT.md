# Base Sepolia Deployment - Complete Setup

## 🎉 Migration Complete!

Successfully migrated from Ethereum Sepolia to Base Sepolia due to Aave market deprecation on Ethereum Sepolia.

## 📝 Deployment Details

**Network:** Base Sepolia (Chain ID: 84532)  
**Deployer:** `0x370e3E98173D667939479373B915BBAB3Eaa029F`  
**Date:** February 12, 2026

### Smart Contracts

| Contract | Address |
|----------|---------|
| **YieldVault** | `0x6DfAeC53c1055424C959d1E825b2EBC1E53b0E8F` |
| **StrategyManager** | `0xeFdAAaBAC2d15EcfD192f12e3b4690d4f81bef2B` |
| **RebalanceExecutor** | `0x3579B973ac55406F52e85e80CfE8EDF5A1Bca1a4` |
| **AaveAdapter** | `0x3dC9A9CaD6D95373E7fCca002bA36eb0581495a6` |
| **UniswapAdapter** | `0xC621A1314348feA6665e5D6AA1aB9C21f3944892` |

### Active Pools

✅ **USDC Pool**
- Token: `0x036CbD53842c5426634e7929541eC2318f3dCF7e`
- aToken: `0xf53B60F4006cab2b3C4688ce41fD5362427A2A66`
- APY: ~1.22%
- Status: Active

✅ **WETH Pool**
- Token: `0x4200000000000000000000000000000000000006`
- aToken: `0x96e32dE4B1d1617B8c2AE13a88B9cC287239b13f`
- APY: ~69.77%
- Status: Active

## 🚀 Next Steps

### 1. Get Testnet Tokens

From the Aave Base Sepolia faucet (https://staging.aave.com/faucet/):
- ✅ USDC: 10,000 tokens available
- ✅ USDT: 10,000 tokens available (alternative)

Or wrap ETH to WETH:
```bash
python scripts/wrap_eth.py --amount 0.01  # Wraps 0.01 ETH to WETH
```

### 2. Check Your Balances

```bash
python scripts/check_balances_base.py
```

### 3. Deposit Tokens into Vault

Once you have tokens from the faucet, deposit them:

```bash
# Example: Deposit 100 USDC
python src/execution/deposit_testnet.py --network base_sepolia --asset USDC --amount 100

# Example: Deposit 0.05 WETH
python src/execution/deposit_testnet.py --network base_sepolia --asset WETH --amount 0.05
```

### 4. Start the Keeper

The keeper has been updated to use Base Sepolia by default:

```bash
python src/execution/keeper_service.py
```

Or with custom interval:
```bash
python src/execution/keeper_service.py --interval 10  # 10-minute intervals
```

## 💡 Why Base Sepolia?

1. **Active Aave Markets**: Unlike Ethereum Sepolia, Base Sepolia has active Aave V3 markets with real APY
2. **Lower Gas Fees**: Base is an L2, offering significantly lower transaction costs
3. **Better Testing**: More realistic testnet environment for DeFi protocols
4. **WETH Support**: 69.77% APY on WETH makes multi-asset testing meaningful

## 🔍 Monitoring

### Check Keeper Status
```bash
# View running processes
ps aux | grep keeper_service

# View logs
tail -f logs/keeper_service.log
```

### Check Predictions
```bash
bash scripts/check_predictions.sh
```

### Check Balances
```bash
python scripts/check_balances_base.py
```

## 📊 Pool Configuration

The keeper monitors both USDC and WETH pools:
- **Prediction Updates**: Every 5-15 minutes
- **Rebalancing**: When conditions are met
- **ML Models**: LSTM (APY prediction) + XGBoost (risk classification)

## ⚙️ Configuration Files

- **Deployment**: `deployments/base_sepolia_deployment.json`
- **Network Config**: `.env` (BASE_SEPOLIA_RPC_URL)
- **ML Models**: `models/` directory (LSTM, XGBoost, scalers)
- **Database**: PostgreSQL `defi_yield_db`

## 🎯 Expected Behavior

1. **Keeper starts** and connects to Base Sepolia
2. **ML predictions** are generated for USDC and WETH pools
3. **Predictions are logged** to database and updated on-chain
4. **Rebalancing executes** when:
   - Cooldown period has passed (5 minutes minimum)
   - New allocation differs significantly from current
   - ML recommends rebalancing
5. **Database tracking** records all predictions and rebalancing events

## 📈 Performance Metrics (from Sepolia)

Previous performance on Ethereum Sepolia:
- 69 predictions logged
- 13 keeper cycles completed
- 92.3% success rate
- Average gas: ~30k per operation

Expected on Base Sepolia:
- **Lower gas costs** (~10x reduction)
- **Faster confirmations**
- **More meaningful APY differences** (WETH 69% vs USDC 1%)

## 🔗 Useful Links

- **Base Sepolia Explorer**: https://sepolia.basescan.org/
- **Aave Faucet**: https://staging.aave.com/faucet/
- **Base Bridge**: https://bridge.base.org/
- **RPC**: Your Alchemy/Infura Base Sepolia endpoint

## ✅ Verification

To verify contracts on BaseScan:
```bash
npm run verify:base_sepolia
```

---

**Status:** ✅ Deployment Complete  
**Ready for:** Token acquisition and keeper operation  
**Next Action:** Get USDC/WETH from faucet, deposit into vault, start keeper
