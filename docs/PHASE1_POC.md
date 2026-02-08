# Phase 1: Proof of Concept (Data & Prediction)
## Duration: Months 1-3 | Status: 🎯 READY TO START

---

## Objective

Validate that machine learning models can accurately predict DeFi protocol APY and identify profitable rebalancing opportunities using historical data. This phase focuses entirely on **off-chain simulation** with no on-chain execution.

## Success Criteria

- [ ] Data pipeline ingests 1M+ data points from 10+ protocols
- [ ] LSTM yield predictor achieves MAPE < 10% on validation set
- [ ] XGBoost risk classifier achieves >80% accuracy
- [ ] Backtesting framework simulates realistic gas costs and slippage
- [ ] ML-based strategy outperforms baseline by >2% APY (net of costs)
- [ ] All code documented and peer-reviewed

---

## Month 1: Data Infrastructure

### Week 1-2: Environment Setup & Data Source Integration

#### Tasks

1. **Development Environment**
   ```bash
   # Set up Python environment
   python3.10 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
   # Install PostgreSQL with TimescaleDB
   # (See installation guide for your OS)
   
   # Set up Redis
   docker run -d -p 6379:6379 redis:7
   ```

2. **The Graph Subgraph Deployment**
   - Deploy subgraphs for Aave V3, Curve, Uniswap V3
   - Configure GraphQL endpoints
   - Test queries for deposit/withdraw events
   
   **Example Query (Aave):**
   ```graphql
   {
     reserves(orderBy: liquidityRate, orderDirection: desc, first: 10) {
       id
       symbol
       liquidityRate
       variableBorrowRate
       totalLiquidity
       availableLiquidity
     }
   }
   ```

3. **Alchemy RPC Setup**
   - Create Alchemy app (Ethereum Mainnet + Goerli)
   - Configure webhook for vault address monitoring
   - Test Enhanced APIs (getTokenBalances, simulateExecution)

4. **Dune Analytics Integration**
   - Create Dune account and generate API key
   - Write queries for:
     - 30-day APY volatility by protocol
     - Whale activity (>$1M deposits)
     - TVL trends
   - Schedule automated query execution

#### Deliverables

- [ ] `src/data/graph_client.py` - The Graph integration
- [ ] `src/data/alchemy_client.py` - Alchemy RPC wrapper
- [ ] `src/data/dune_client.py` - Dune Analytics fetcher
- [ ] `db/schema.sql` - TimescaleDB schema
- [ ] `.env` configured with all API keys
- [ ] Test scripts verify all data sources responding

### Week 3-4: Data Pipeline & Feature Engineering

#### Tasks

1. **Data Aggregation Service**
   
   Create `src/data/ingestion_service.py`:
   ```python
   class DataAggregator:
       def __init__(self):
           self.graph = GraphClient()
           self.alchemy = AlchemyClient()
           self.dune = DuneClient()
           self.db = TimescaleDB()
           self.cache = Redis()
       
       async def collect_protocol_data(self, protocol_id):
           # Fetch from multiple sources
           # Normalize and validate
           # Store in database + cache
           pass
   ```

2. **Feature Engineering Pipeline**
   
   Implement 32-dimensional feature vector:
   - **Temporal:** APY, rolling means, volatility, time encoding
   - **Protocol Health:** TVL, utilization, liquidity depth
   - **Market Dynamics:** Gas prices, trading volume, whale activity
   - **Competitor Signals:** Relative APY, TVL migration
   
   Create `src/data/feature_engineering.py`

3. **Historical Data Collection**
   
   Backfill 18 months of data (2024-01-01 to 2025-06-30):
   ```bash
   python scripts/backfill_historical_data.py \
     --start 2024-01-01 \
     --end 2025-06-30 \
     --protocols aave,curve,uniswap
   ```

4. **Data Quality Checks**
   - Missing value handling
   - Outlier detection (Z-score > 4)
   - Data consistency validation
   - Generate data quality report

#### Deliverables

- [ ] `src/data/ingestion_service.py` - Main aggregator
- [ ] `src/data/feature_engineering.py` - Feature pipeline
- [ ] `scripts/backfill_historical_data.py` - Data collection script
- [ ] Historical dataset: 18 months, 10+ protocols, 1M+ rows
- [ ] Data quality report: `data/quality_report.pdf`

---

## Month 2: ML Model Development

### Week 5-6: LSTM Yield Prediction Model

#### Model Architecture

```python
class YieldForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=128, 
                             num_layers=2, dropout=0.2)
        self.lstm2 = nn.LSTM(hidden_size=128, hidden_size=64)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        self.fc = nn.Linear(64, 1)
        
    def forward(self, x):
        # x: (batch, seq_len=30, features=32)
        lstm_out, _ = self.lstm1(x)
        lstm_out, _ = self.lstm2(lstm_out)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return self.fc(attn_out[:, -1, :])
```

#### Training Strategy

1. **Dataset Split**
   - Training: 2024-01-01 to 2025-03-31 (15 months)
   - Validation: 2025-04-01 to 2025-05-31 (2 months)
   - Test: 2025-06-01 to 2025-06-30 (1 month)

2. **Hyperparameters**
   ```python
   config = {
       'learning_rate': 1e-3,
       'batch_size': 64,
       'num_epochs': 100,
       'lookback_window': 30,  # days
       'prediction_horizon': 7,  # days
       'early_stopping_patience': 10
   }
   ```

3. **Loss Function**
   ```python
   loss = 0.7 * mape_loss + 0.3 * direction_loss
   # Penalize wrong directional predictions
   ```

4. **Walk-Forward Validation**
   - Retrain monthly on expanding window
   - Prevents look-ahead bias

#### Tasks

- [ ] Implement LSTM architecture in `src/ml/yield_predictor.py`
- [ ] Create data loaders with proper windowing
- [ ] Set up TensorBoard logging
- [ ] Train model on GPU (use PyTorch Lightning)
- [ ] Generate validation metrics (MAPE, R², directional accuracy)
- [ ] Save model checkpoints to `models/lstm_yield_predictor_v1.pth`

#### Target Metrics

| Protocol | MAPE Target | Actual |
|----------|-------------|--------|
| Aave USDC | <5% | TBD |
| Curve 3pool | <8% | TBD |
| Uniswap USDC/DAI | <15% | TBD |

### Week 7-8: XGBoost Risk Classification Model

#### Model Configuration

```python
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,  # Low, Medium, High risk
    max_depth=8,
    learning_rate=0.05,
    n_estimators=500,
    eval_metric='mlogloss'
)
```

#### Feature Engineering (45 dimensions)

Create `src/ml/risk_features.py`:

1. **Smart Contract Security (15 features)**
   - Audit scores (Certik, Trail of Bits)
   - Bug bounty program metrics
   - Historical exploit count
   - Code complexity metrics

2. **Protocol Maturity (10 features)**
   - Days since deployment
   - Cumulative TVL-days
   - User diversity (Gini coefficient)

3. **Economic Risks (12 features)**
   - Impermanent loss history
   - Oracle reliability
   - Collateral diversity

4. **Operational Risks (8 features)**
   - Exit liquidity
   - Centralization score
   - Regulatory scrutiny

#### Labeling Strategy

Create training labels:
- **Low Risk (0-30):** Aave V3, Curve (major pools), Uniswap V3 (blue-chip pairs)
- **Medium Risk (31-70):** Newer protocols with 1+ audits
- **High Risk (71-100):** Unaudited, exploit history, or anonymous teams

Manually label 500 protocols, then use semi-supervised learning

#### Tasks

- [ ] Collect risk features for 500+ protocols
- [ ] Expert labeling session (invite security auditors)
- [ ] Implement XGBoost training pipeline
- [ ] Use SHAP for feature importance analysis
- [ ] Cross-validation with time-series split
- [ ] Generate confusion matrix and classification report
- [ ] Save model to `models/xgboost_risk_scorer_v1.pkl`

#### Target Metrics

- Accuracy: >80%
- Precision (High Risk): >90% (avoid false positives)
- Recall (High Risk): >70%
- AUC-ROC: >0.85

---

## Month 3: Backtesting & Validation

### Week 9-10: Custom Backtesting Engine

#### Architecture

```python
class DeFiBacktester:
    def __init__(self, start_date, end_date, initial_capital):
        self.historical_data = self.load_data(start_date, end_date)
        self.portfolio = Portfolio(initial_capital)
        self.risk_engine = RiskEngine()
        self.ml_models = self.load_models()
        
    def run(self, strategy):
        for timestamp in self.historical_data.index:
            state = self.get_state(timestamp)
            predictions = self.ml_models.predict(state)
            action = strategy.decide(state, predictions)
            result = self.simulate_execution(action, timestamp)
            self.portfolio.update(result)
```

#### Realistic Simulation Components

1. **Gas Cost Model**
   ```python
   def estimate_gas_cost(action, gas_price_gwei):
       base_gas = 150_000  # Base transaction
       per_protocol_gas = 80_000  # Per protocol interaction
       
       num_protocols = count_protocol_changes(action)
       total_gas = base_gas + (num_protocols * per_protocol_gas)
       
       cost_eth = (total_gas * gas_price_gwei) / 1e9
       cost_usd = cost_eth * eth_price
       return cost_usd
   ```

2. **Slippage Model**
   ```python
   def calculate_slippage(trade_size, liquidity):
       # AMM constant product formula
       impact = trade_size / liquidity
       slippage = impact ** 2  # Quadratic for large trades
       return min(slippage, 0.05)  # Cap at 5%
   ```

3. **Protocol-Specific Mechanics**
   - Aave: Variable interest rates, utilization-based
   - Curve: Virtual price adjustments, A-parameter
   - Uniswap V3: Concentrated liquidity, fee tiers

#### Tasks

- [ ] Implement `src/backtesting/engine.py`
- [ ] Create realistic gas cost simulator
- [ ] Model slippage for each protocol
- [ ] Implement portfolio accounting (ERC4626-like)
- [ ] Add transaction logging and replay capability
- [ ] Create visualization dashboard (Plotly)

### Week 11-12: Strategy Backtesting & Analysis

#### Baseline Strategies

1. **Buy-and-Hold:** Equal-weight allocation, rebalance never
2. **Periodic Rebalancing:** Monthly rebalance to equal-weight
3. **Simple Rule-Based:** Allocate to top-3 APY protocols

#### ML Strategies

1. **LSTM-Only:** Use predicted APY, ignore risk
2. **Risk-Filtered:** Only allocate to protocols with risk < 50
3. **Full ML:** LSTM predictions + XGBoost risk filtering

#### Backtest Execution

```bash
python scripts/run_backtest.py \
  --strategy ml_full \
  --start 2024-01-01 \
  --end 2025-12-31 \
  --capital 1000000 \
  --rebalance-freq weekly
```

#### Analysis Metrics

```python
def analyze_results(backtest_df):
    returns = backtest_df['portfolio_value'].pct_change()
    
    metrics = {
        'total_return': final_value / initial_value - 1,
        'cagr': calculate_cagr(returns),
        'sharpe_ratio': returns.mean() / returns.std() * sqrt(365),
        'sortino_ratio': downside_adjusted_sharpe(returns),
        'max_drawdown': calculate_max_drawdown(backtest_df),
        'calmar_ratio': cagr / abs(max_drawdown),
        'win_rate': (returns > 0).sum() / len(returns),
        'total_gas_spent': backtest_df['gas_cost'].sum(),
        'num_rebalances': count_rebalances(backtest_df),
        'avg_holding_period': calculate_avg_holding_period(backtest_df)
    }
    
    return metrics
```

#### Tasks

- [ ] Implement baseline strategies
- [ ] Run backtests for all strategies (5 total)
- [ ] Generate comparative analysis report
- [ ] Create visualizations:
   - Portfolio value over time
   - Drawdown chart
   - Allocation heatmap
   - Risk score evolution
- [ ] Statistical significance testing (t-test)
- [ ] Sensitivity analysis (vary gas prices, slippage)

#### Target Results

| Strategy | Target APY | Target Sharpe | Target Max DD |
|----------|-----------|---------------|---------------|
| Buy-and-Hold | 4.5% | 1.8 | <12% |
| Periodic | 5.0% | 1.9 | <10% |
| Rule-Based | 5.5% | 2.0 | <10% |
| **ML Full** | **>7.0%** | **>2.2** | **<8%** |

---

## Deliverables Checklist

### Code

- [ ] `src/data/graph_client.py` - The Graph integration
- [ ] `src/data/alchemy_client.py` - Alchemy RPC wrapper
- [ ] `src/data/dune_client.py` - Dune Analytics client
- [ ] `src/data/ingestion_service.py` - Data aggregator
- [ ] `src/data/feature_engineering.py` - Feature pipeline
- [ ] `src/ml/yield_predictor.py` - LSTM model
- [ ] `src/ml/risk_scorer.py` - XGBoost classifier
- [ ] `src/backtesting/engine.py` - Backtesting framework
- [ ] `src/backtesting/strategies.py` - Strategy implementations
- [ ] `scripts/backfill_historical_data.py` - Data collection
- [ ] `scripts/train_models.py` - Model training script
- [ ] `scripts/run_backtest.py` - Backtest execution

### Data

- [ ] Historical dataset (18 months, 10+ protocols, 1M+ rows)
- [ ] Trained LSTM model: `models/lstm_yield_predictor_v1.pth`
- [ ] Trained XGBoost model: `models/xgboost_risk_scorer_v1.pkl`
- [ ] Feature importance analysis
- [ ] Data quality report

### Documentation

- [ ] **Backtest Report** (50+ pages):
  - Executive summary
  - Methodology
  - Model architectures and training
  - Strategy descriptions
  - Results and metrics
  - Comparative analysis
  - Sensitivity analysis
  - Limitations and assumptions
  - Recommendations for Phase 2
- [ ] Code documentation (docstrings, README)
- [ ] API documentation (if exposing APIs)
- [ ] Jupyter notebooks with exploratory analysis

### Presentations

- [ ] Stakeholder presentation (15 slides):
  - Problem statement
  - Approach
  - Key results
  - Next steps
- [ ] Technical deep-dive (30 slides, for team)

---

## Key Decisions to Make

### Decision 1: Which protocols to prioritize?

**Options:**
- **Option A:** Focus on 3 major protocols (Aave, Curve, Uniswap)
- **Option B:** Include 10+ protocols for diversity

**Recommendation:** Start with Option A (3 protocols) for Phase 1 simplicity, expand in Phase 2.

### Decision 2: Rebalancing frequency?

**Options:**
- Daily: More opportunities, higher gas costs
- Weekly: Balanced approach
- Monthly: Lower costs, fewer opportunities

**Recommendation:** Backtest all three, likely settle on **weekly** for optimal gas efficiency.

### Decision 3: Train one model per protocol or unified model?

**Options:**
- **Option A:** One LSTM per protocol (3 models)
- **Option B:** Single LSTM with protocol embedding

**Recommendation:** Option B (unified model) for transfer learning benefits.

---

## Risk Management (Phase 1)

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Data quality issues | Medium | High | Extensive validation, multiple sources |
| Model overfitting | High | Medium | Cross-validation, regularization, walk-forward |
| API rate limits | Medium | Low | Caching, request throttling |
| Infrastructure costs | Low | Low | Use spot instances, optimize queries |

### Timeline Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Data collection delays | Medium | Medium | Start early, parallelize |
| Model training time | Low | Low | Use GPU instances |
| Scope creep | High | High | Strict milestone tracking |

---

## Success Metrics Summary

✅ **Phase 1 Complete When:**
- All code peer-reviewed and documented
- LSTM MAPE < 10% on test set
- XGBoost accuracy > 80%
- ML strategy outperforms baseline by >2% APY
- Comprehensive backtest report delivered
- Stakeholder approval to proceed to Phase 2

---

## Estimated Costs (Phase 1)

| Item | Cost |
|------|------|
| Alchemy API (3 months) | $300 |
| The Graph (hosted service) | $0 (free tier) |
| Dune Analytics API | $200/month = $600 |
| AWS (RDS, EC2, S3) | $500/month = $1,500 |
| GPU compute (training) | $300 |
| Team salaries (4 people) | *Variable* |
| **Total Infrastructure** | **~$2,700** |

---

## Next Steps After Phase 1

1. **Review Results:** Convene team to analyze backtest report
2. **Go/No-Go Decision:** Determine if accuracy justifies Phase 2 investment
3. **Refine Models:** Address any weaknesses identified
4. **Plan Phase 2:** Kick off risk engine development

---

**Status:** Ready to begin  
**Owner:** ML Engineering Team  
**Last Updated:** February 8, 2026
