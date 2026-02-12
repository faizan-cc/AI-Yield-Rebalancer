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

### Week 2 - Days 4-5: Paper Trading & Data Validation
- [ ] **Data Pipeline Validation:**
  - [ ] Compare testnet APYs to mainnet APYs (should be within 2x)
  - [ ] Verify testnet Aave USDC APY matches Aave docs
  - [ ] Check if DefiLlama has testnet data (implement fallback if not)
  - [ ] Validate ML model receives correct inputs
  - [ ] Test data ingestion every 5 minutes for 24 hours
- [ ] **Paper Trading:**
  - [ ] Run ML models on live testnet data
  - [ ] Generate rebalancing proposals (no execution)
  - [ ] Compare paper trading vs backtest using strategy_validator.py
  - [ ] Validate slippage estimates
  - [ ] Track prediction accuracy (APY predictions vs actual)

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
- [ ] **Deploy 4 independent vault instances:**
  - [ ] Vault A: Strategy 1 (Highest APY) - $300
  - [ ] Vault B: Strategy 2 (TVL Weighted) - $200 ⚠️ DON'T SKIP!
  - [ ] Vault C: Strategy 3 (Optimized ML) - $500
  - [ ] Vault D: Strategy 4 (Stablecoin) - $200
- [ ] **Setup fair compa & Stress Testing

### Week 4 - Days 1-3: Performance Analysis & Stress Tests
- [ ] **Performance Validation:**
  - [ ] Run strategy_validator.py comprehensive report
  - [ ] Calculate actual returns vs backtest (must be within ±50%)
  - [ ] Measure actual slippage vs predicted (must be <0.5%)
  - [ ] Gas cost analysis per rebalance (must be <0.01 ETH)
  - [ ] Identify bottlenecks and optimization opportunities
- [ ] **Stress Testing Scenarios:**
  - [ ] Test during testnet gas spikes (>100 gwei)
  - [ ] Simulate USDC depeg (mock oracle to $0.97)
  - [ ] Test when Aave is at 95% utilization
  - [ ] Force high slippage (large swap on low liquidity)
  - [ ] Test multi-block rebalancing (pending tx >5 blocks)
  - [ ] Simulate keeper downtime (miss 1-2 windows)
- [ ] **Adversarial Testing:**
  - [ ] Test kill switch with simulated depeg
  - [ ] Verify emergency withdrawal works
  - [ ] Test recovery after crisis
  - [ ] Test with stale oracle data (>1 hour old)
  - [ ] Test frontrunning scenariolidator.py for each vault
  - [ ] Enable real-time deviation alerts (>20% from backtest)
  - [ ] Start 24/7 keeper service
Final Validation & Go/No-Go Decision
- [ ] **Generate Comprehensive Reports:**
  - [ ] Run strategy_validator.py final report
  - [ ] Export all metrics to CSV
  - [ ] Create performance comparison charts
  - [ ] Document all anomalies and resolutions
- [ ] **Complete Go/No-Go Checklist** (see below)
- [ ] **Documentation:**
  - [ ] Write deployment guide
  - [ ] Create user documentation
  - [ ] Prepare mainnet checklist
  - [ ] Document lessons learned
- [ ] **Next Steps:**
  - [ ] If APPROVED: Schedule professional audit (Trail of Bits)
  - [ ] If REJECTED: Document failures and create fix plan backtest
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
- [🚦 GO/NO-GO DECISION CRITERIA (Week 4, Day 7)

### Performance Validation ✅
- [ ] **Testnet return within ±50% of backtest (+7.90% ±4%)**
  - Actual: _____% | Expected: 3.95% to 11.85% | Status: ☐ PASS ☐ FAIL
  
- [ ] **Sharpe ratio within ±30% of backtest (22.0 ±6.6)**
  - Actual: _____ | Expected: 15.4 to 28.6 | Status: ☐ PASS ☐ FAIL
  
- [ ] **Max drawdown < backtest max drawdown (-0.11%)**
  - Actual: _____% | Must be better than -0.11% | Status: ☐ PASS ☐ FAIL

### Operational Validation ✅
- [ ] **Average gas cost < 0.01 ETH per rebalance**
  - Actual: _____ ETH | Status: ☐ PASS ☐ FAIL
  
- [ ] **Worst-case slippage < 0.5% on all trades**
  - Actual: _____% | Status: ☐ PASS ☐ FAIL
  
- [ ] **System uptime > 95% over 4 weeks**
  - Actual: _____% | Max downtime: 33.6 hours | Status: ☐ PASS ☐ FAIL
  
- [ ] **Zero failed transactions (all reverts debugged)**
  - Failed tx count: _____ | Status: ☐ PASS ☐ FAIL
  
- [ ] **Keeper service reliability > 98%**
  - Successful rebalances: ____ / ____ | Status: ☐ PASS ☐ FAIL

### Strategy Comparison ✅
- [ ] **ML strategy outperforms TVL Weighted baseline**
  - ML return: _____% | TVL return: _____% | Spread: _____% (>1% required)
  - Status: ☐ PASS ☐ FAIL
  
- [ ] **All 4 strategies tested simultaneously for 2+ weeks**
  - Status: ☐ PASS ☐ FAIL
  
- [ ] **No strategy loses >10% of capital**
  - Strategy 1: _____% | Strategy 2: _____% | Strategy 3: _____% | Strategy 4: _____%
  - Status: ☐ PASS ☐ FAIL

### Security & Robustness ✅
- [ ] **Internal security audit completed**
  - Status: ☐ PASS ☐ FAIL
  � CONTINUOUS OPERATION LOG (Week 3-4)

Track daily for 4-week continuous test:

| Date | Rebalances | Gas (ETH) | Slippage | Portfolio Value | Uptime % | Issues |
|------|------------|-----------|----------|-----------------|----------|--------|
| 2026-__-__ | __/4 | _____ | _____% | $_____ | _____% | _____ |
| 2026-__-__ | __/4 | _____ | _____% | $_____ | _____% | _____ |
| ... | ... | ... | ... | ... | ... | ... |

**Total Rebalances Expected:** 112 (4 per day × 28 days)  
**Actual Successful:** _____ / 112 (target: >106 for 95% uptime)

---

## ⚠️ ANOMALY LOG

Track all unexpected behaviors:

| Date | Severity | Issue | Impact | Resolution | Status |
|------|----------|-------|--------|------------|--------|
| 2026-__-__ | 🔴 Critical | _____ | _____ | _____ | ☐ Open ☐ Resolved |
| 2026-__-__ | 🟡 Medium | _____ | _____ | _____ | ☐ Open ☐ Resolved |

---

## 📞 Next Actions (Today)

### Immediate (Day 1)
1. ✅ **Add testnet RPC URLs to .env**
2. ✅ **Get testnet ETH from faucets**
3. ✅ **Create strategy_validator.py**
4. ✅ **Add testnet validation page to dashboard**

### This Week (Days 2-7)
5. **Finish remaining smart contracts** (StrategyManager, RebalanceExecutor, Adapters)
6. **Write comprehensive unit tests**
7. **Run initial compilation: `npx hardhat compile`**
8. **Deploy to local Hardhat network first**

---

**Estimated Total Time**: 4 weeks  
**Current Phase**: Phase 1 - Day 1 ✨  
**Success Rate Target**: 8.0/10 final score  
**Last Updated**: 2026-02-11
- [ ] **Kill switch triggers correctly in depeg scenario**
  - Status: ☐ PASS ☐ FAIL

### Data Quality & Prediction Accuracy ✅
- [ ] **ML predictions match reality within 10% error**
  - Average prediction error: _____% | Status: ☐ PASS ☐ FAIL
  
- [ ] **APY data source uptime > 99%**
  - DefiLlama uptime: _____% | Status: ☐ PASS ☐ FAIL
  
- [ ] **No data pipeline failures for 48+ hours**
  - Status: ☐ PASS ☐ FAIL

---

## 📊 FINAL DECISION MATRIX

| Category | Weight | Score (1-10) | Weighted Score |
|----------|--------|--------------|----------------|
| **Performance** | 30% | _____ | _____ |
| **Operational** | 25% | _____ | _____ |
| **Strategy Comparison** | 20% | _____ | _____ |
| **Security & Robustness** | 15% | _____ | _____ |
| **Data Quality** | 10% | _____ | _____ |
| **TOTAL** | **100%** | **_____** | **_____/10** |

### Decision Threshold
- **Score ≥ 8.0/10 AND all critical items PASS:** ✅ **APPROVED FOR MAINNET**
- **Score 6.0-7.9/10:** ⚠️ **CONDITIONAL APPROVAL** (fix minor issues first)
- **Score < 6.0/10 OR any critical FAIL:** ❌ **REJECTED** (major rework needed)

### Final Decision: ☐ APPROVED ☐ CONDITIONAL ☐ REJECTED

**Approver Signature:** _________________ **Date:** _________________

---

## 🚀 POST-DECISION ACTIONS

### If APPROVED ✅
1. Schedule professional audit ($15K-25K budget)
   - Preferred: Trail of Bits or OpenZeppelin
   - Timeline: 2-3 weeks
2. Deploy to mainnet with $1,000 initial capital
3. Run mainnet soft launch for 2 weeks
4. Scale to $10K if no issues
5. Public launch Month 3

### If CONDITIONAL ⚠️
1. Document all items requiring fixes
2. Create 1-week sprint to resolve issues
3. Re-test specific failure areas
4. Re-run go/no-go decision

### If REJECTED ❌
1. Comprehensive post-mortem analysis
2. Identify root causes of failures
3. Implement fixes (2-4 weeks)
4. Restart testnet testing from Phase 3
5. Re-evaluate timeline and approach
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
