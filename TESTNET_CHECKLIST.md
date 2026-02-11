# Testnet Deployment Checklist

## 🚀 Phase 1: Infrastructure Setup

### Week 1 - Days 1-2: Environment Setup
- [ ] Install Hardhat: `npm install`
- [ ] Add testnet RPC endpoints to .env
  - [ ] SEPOLIA_RPC_URL (Alchemy/Infura)
  - [ ] BASE_SEPOLIA_RPC_URL
  - [ ] DEPLOYER_PRIVATE_KEY (testnet wallet)
- [ ] Acquire testnet tokens
  - [ ] Sepolia ETH: https://sepoliafaucet.com
  - [ ] Base Sepolia ETH: https://www.coinbase.com/faucets/base-ethereum-sepolia-faucet
  - [ ] Testnet USDC, WETH, DAI from Aave faucet

### Week 1 - Days 3-5: Smart Contract Development
- [x] YieldVault.sol created (core vault)
- [ ] StrategyManager.sol
- [ ] RebalanceExecutor.sol  
- [ ] AaveAdapter.sol
- [ ] UniswapAdapter.sol
- [ ] MorphoAdapter.sol (if available on testnet)
- [ ] Write unit tests
- [ ] Run gas optimization

### Week 1 - Days 6-7: Initial Deployment
- [ ] Compile contracts: `npx hardhat compile`
- [ ] Deploy to Sepolia
- [ ] Deploy to Base Sepolia
- [ ] Verify contracts on Etherscan/Basescan
- [ ] Test basic deposit/withdraw

## 🔗 Phase 2: ML Integration

### Week 2 - Days 1-3: Bridge Development
- [x] contract_manager.py created
- [ ] strategy_executor.py - Execute ML predictions
- [ ] gas_optimizer.py - Estimate optimal gas
- [ ] rebalancer.py - Automated rebalancing
- [ ] Test off-chain → on-chain flow

### Week 2 - Days 4-5: Paper Trading
- [ ] Run ML models on live testnet data
- [ ] Generate rebalancing proposals (no execution)
- [ ] Compare paper trading vs backtest
- [ ] Validate slippage estimates

### Week 2 - Days 6-7: Integration Testing
- [ ] End-to-end test: ML prediction → contract execution
- [ ] Test emergency pause/unpause
- [ ] Test fee collection
- [ ] Monitor gas costs

## 📊 Phase 3: Live Testnet

### Week 3 - Days 1-2: Fund & Initialize
- [ ] Transfer testnet tokens to vault
- [ ] Add supported assets (USDC, WETH, DAI)
- [ ] Make first deposit
- [ ] Verify share calculation

### Week 3 - Days 3-5: First Rebalancing
- [ ] Run ML strategy (Enhanced ML v3.0)
- [ ] Generate allocation targets
- [ ] Execute first rebalance
- [ ] Monitor 24h for issues

### Week 3 - Days 6-7: Multi-Strategy Deployment
- [ ] Deploy Strategy 1 (Highest APY) with $300
- [ ] Deploy Strategy 3 (ML) with $500
- [ ] Deploy Strategy 4 (Stablecoin) with $200
- [ ] Track all strategies in parallel

## ✅ Phase 4: Validation

### Week 4 - Days 1-3: Performance Analysis
- [ ] Calculate actual returns vs backtest
- [ ] Measure slippage impact
- [ ] Gas cost analysis per rebalance
- [ ] Identify bottlenecks

### Week 4 - Days 4-5: Security Review
- [ ] Internal security audit
- [ ] Test attack vectors
- [ ] Validate access controls
- [ ] Document vulnerabilities found

### Week 4 - Days 6-7: Documentation
- [ ] Write deployment guide
- [ ] Create user documentation
- [ ] Prepare mainnet checklist
- [ ] Schedule professional audit

## 📋 Required Environment Variables

Add to `.env`:
```bash
# Testnet RPCs
SEPOLIA_RPC_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_KEY
BASE_SEPOLIA_RPC_URL=https://base-sepolia.g.alchemy.com/v2/YOUR_KEY

# Deployer wallet (TESTNET ONLY - never use mainnet keys!)
DEPLOYER_PRIVATE_KEY=0x...

# API Keys
ETHERSCAN_API_KEY=...
BASESCAN_API_KEY=...
COINMARKETCAP_API_KEY=... (for gas reporting)

# Testnet Protocol Addresses (to be added after lookup)
SEPOLIA_AAVE_POOL=0x...
SEPOLIA_UNISWAP_ROUTER=0x...
BASE_SEPOLIA_AAVE_POOL=0x...
BASE_SEPOLIA_UNISWAP_ROUTER=0x...
```

## 🎯 Success Metrics

### Must-Have (Go/No-Go)
- [ ] All contracts deployed without errors
- [ ] At least 1 successful rebalance
- [ ] No critical security issues
- [ ] Deposit/withdraw working correctly

### Nice-to-Have
- [ ] Gas costs < 0.01 ETH per rebalance
- [ ] Returns within 50% of backtest
- [ ] 4 weeks continuous operation
- [ ] Multiple strategies tested

## 🚨 Risk Management

### Before Each Deployment
- [ ] Test on local hardhat network first
- [ ] Verify contract code on block explorer
- [ ] Start with small amounts ($100-500)
- [ ] Have emergency pause plan ready

### Emergency Procedures
1. If critical bug found: Call `pause()` immediately
2. If funds stuck: Use admin withdraw functions
3. If attack detected: Pause + notify team
4. Document all incidents for audit

## 📞 Next Actions (Today)

1. **Add testnet RPC URLs to .env**
2. **Get testnet ETH from faucets**
3. **Finish remaining smart contracts**
4. **Run initial compilation test**

---

**Estimated Total Time**: 4 weeks
**Current Phase**: Phase 1 - Day 1 ✨
