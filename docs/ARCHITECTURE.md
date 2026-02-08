# System Architecture Documentation

## 1. Overview

The AI-Driven DeFi Yield Rebalancing System follows a **4-layer architecture** with clear separation between data ingestion, risk assessment, AI inference, and on-chain execution. This document provides detailed specifications for component interactions, data flow, and communication protocols.

## 2. Architecture Layers

### Layer 1: Data Ingestion

**Purpose:** Aggregate real-time and historical blockchain data from multiple sources

**Components:**
- The Graph Subgraph Clients
- Alchemy RPC Interface
- Dune Analytics Historical Data Fetcher
- Data Normalization Pipeline
- TimescaleDB Storage

**Data Flow:**
```
External Sources → Data Aggregator → Normalization → TimescaleDB + Redis Cache
```

### Layer 2: Risk Assessment Engine

**Purpose:** Score protocol safety and monitor kill-switch triggers

**Components:**
- Multi-Dimensional Risk Scorer
- Chainlink Oracle Monitor
- Liquidity Depth Analyzer
- Anomaly Detector (Isolation Forest + Autoencoder)
- Kill-Switch Manager

**Input:** Normalized protocol data from Layer 1  
**Output:** Risk scores (0-100) + boolean kill-switch flags

### Layer 3: AI Inference Engine

**Purpose:** Predict yields and generate optimal rebalancing decisions

**Components:**
- LSTM Yield Predictor (PyTorch)
- XGBoost Risk Classifier
- Transformer Systemic Risk Model
- PPO Rebalancing Agent (Stable-Baselines3)
- Gas Optimizer

**Input:** Risk-scored data from Layer 2  
**Output:** Target allocation vector + confidence scores

### Layer 4: Execution Layer

**Purpose:** Execute rebalancing transactions on-chain

**Components:**
- Smart Contracts (Vault, StrategyHub, KillSwitch)
- Keeper Service (transaction signing + broadcasting)
- Protocol Adapters (Aave, Curve, Uniswap)
- Multi-Sig Governance (Gnosis Safe)

**Input:** Approved allocations from Layer 3  
**Output:** On-chain state changes + events

## 3. Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 1: DATA INGESTION                          │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │  The Graph   │  │   Alchemy    │  │     Dune     │            │
│  │  Subgraphs   │  │   RPC API    │  │  Analytics   │            │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘            │
│         │                  │                  │                     │
│         └──────────────────┼──────────────────┘                     │
│                            │                                        │
│                   ┌────────▼────────┐                              │
│                   │ Data Aggregator │                              │
│                   │  (FastAPI)      │                              │
│                   └────────┬────────┘                              │
│                            │                                        │
│            ┌───────────────┼───────────────┐                       │
│            │               │               │                       │
│    ┌───────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐              │
│    │ TimescaleDB  │ │   Redis     │ │  Feature    │              │
│    │  (Storage)   │ │  (Cache)    │ │ Engineering │              │
│    └───────┬──────┘ └──────┬──────┘ └──────┬──────┘              │
└────────────┼────────────────┼────────────────┼──────────────────────┘
             │                │                │
             │                │                │
┌────────────┼────────────────┼────────────────┼──────────────────────┐
│            │    LAYER 2: RISK ASSESSMENT ENGINE                     │
│            │                │                │                       │
│    ┌───────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐              │
│    │ Risk Scorer  │ │  Chainlink  │ │  Anomaly    │              │
│    │  (XGBoost)   │ │   Monitor   │ │  Detector   │              │
│    └───────┬──────┘ └──────┬──────┘ └──────┬──────┘              │
│            │                │                │                       │
│            └────────────────┼────────────────┘                       │
│                             │                                        │
│                   ┌─────────▼─────────┐                            │
│                   │ Kill-Switch       │                            │
│                   │ Orchestrator      │                            │
│                   └─────────┬─────────┘                            │
└─────────────────────────────┼──────────────────────────────────────┘
                              │ Risk-Scored Data
                              │
┌─────────────────────────────┼──────────────────────────────────────┐
│               LAYER 3: AI INFERENCE ENGINE                         │
│                             │                                        │
│              ┌──────────────▼──────────────┐                       │
│              │    State Encoder            │                       │
│              │  (Transforms data to 85-dim)│                       │
│              └──────────────┬──────────────┘                       │
│                             │                                        │
│         ┌───────────────────┼───────────────────┐                  │
│         │                   │                   │                  │
│  ┌──────▼──────┐  ┌─────────▼────────┐  ┌──────▼──────┐          │
│  │ LSTM Yield  │  │  Transformer     │  │  XGBoost    │          │
│  │ Predictor   │  │ Systemic Risk    │  │ Risk Class  │          │
│  └──────┬──────┘  └─────────┬────────┘  └──────┬──────┘          │
│         │                   │                   │                  │
│         └───────────────────┼───────────────────┘                  │
│                             │                                        │
│                   ┌─────────▼─────────┐                            │
│                   │   PPO RL Agent    │                            │
│                   │  (Rebalancer)     │                            │
│                   └─────────┬─────────┘                            │
│                             │                                        │
│                   ┌─────────▼─────────┐                            │
│                   │  Gas Optimizer    │                            │
│                   │  (EIP-1559)       │                            │
│                   └─────────┬─────────┘                            │
└─────────────────────────────┼──────────────────────────────────────┘
                              │ Target Allocations
                              │
┌─────────────────────────────┼──────────────────────────────────────┐
│                 LAYER 4: EXECUTION                                 │
│                             │                                        │
│              ┌──────────────▼──────────────┐                       │
│              │  Transaction Builder        │                       │
│              │  (Constructs calldata)      │                       │
│              └──────────────┬──────────────┘                       │
│                             │                                        │
│              ┌──────────────▼──────────────┐                       │
│              │  Tenderly Simulation        │                       │
│              │  (Pre-flight check)         │                       │
│              └──────────────┬──────────────┘                       │
│                             │                                        │
│              ┌──────────────▼──────────────┐                       │
│              │  HSM Signer                 │                       │
│              │  (AWS KMS / Ledger)         │                       │
│              └──────────────┬──────────────┘                       │
│                             │                                        │
│              ┌──────────────▼──────────────┐                       │
│              │  Keeper Service             │                       │
│              │  (Broadcasts tx)            │                       │
│              └──────────────┬──────────────┘                       │
│                             │                                        │
│                             │ Via Flashbots                         │
│                             │                                        │
│         ┌───────────────────▼───────────────────┐                  │
│         │        ETHEREUM MAINNET                │                  │
│         │                                        │                  │
│         │  ┌────────────────────────────────┐  │                  │
│         │  │  Vault (ERC4626)               │  │                  │
│         │  │  - User deposits               │  │                  │
│         │  │  - Share accounting            │  │                  │
│         │  └───────────┬────────────────────┘  │                  │
│         │              │                        │                  │
│         │  ┌───────────▼────────────────────┐  │                  │
│         │  │  StrategyHub                   │  │                  │
│         │  │  - Rebalancing logic           │  │                  │
│         │  │  - Protocol adapters           │  │                  │
│         │  └───────────┬────────────────────┘  │                  │
│         │              │                        │                  │
│         │  ┌───────────▼────────────────────┐  │                  │
│         │  │  KillSwitch Manager            │  │                  │
│         │  │  - Safety checks               │  │                  │
│         │  │  - Emergency withdrawal        │  │                  │
│         │  └────────────────────────────────┘  │                  │
│         │                                        │                  │
│         │  ┌────────────────────────────────┐  │                  │
│         │  │  Protocol Integrations         │  │                  │
│         │  │  - Aave V3                     │  │                  │
│         │  │  - Curve Finance               │  │                  │
│         │  │  - Uniswap V3                  │  │                  │
│         │  └────────────────────────────────┘  │                  │
│         └────────────────────────────────────────┘                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 4. Communication Protocols

### 4.1 Layer 1 → Layer 2 (Data → Risk)

**Protocol:** gRPC  
**Format:** Protocol Buffers  
**Port:** 50051

**Message Schema:**
```protobuf
message ProtocolData {
  string protocol_id = 1;
  int64 timestamp = 2;
  double current_apy = 3;
  double tvl_usd = 4;
  double utilization_rate = 5;
  double liquidity_depth = 6;
  repeated Transaction recent_transactions = 7;
}

message RiskRequest {
  repeated ProtocolData protocols = 1;
}

message RiskResponse {
  repeated ProtocolRiskScore scores = 1;
  bool any_kill_switch_triggered = 2;
  repeated string active_alerts = 3;
}
```

### 4.2 Layer 2 → Layer 3 (Risk → AI)

**Protocol:** REST API (HTTPS)  
**Format:** JSON  
**Port:** 8000

**Endpoint:** `POST /api/v1/inference/predict`

**Request:**
```json
{
  "timestamp": "2026-02-08T12:00:00Z",
  "protocols": [
    {
      "id": "aave_usdc",
      "current_apy": 0.082,
      "risk_score": 25,
      "tvl_usd": 5000000000,
      "liquidity_depth": 100000000,
      "features": [...]  // 32-dim feature vector
    }
  ],
  "portfolio_state": {
    "value_usd": 1000000,
    "current_allocations": {"aave_usdc": 0.45, ...}
  }
}
```

**Response:**
```json
{
  "predictions": {
    "aave_usdc_7d_apy": 0.078,
    "curve_3pool_7d_apy": 0.052
  },
  "recommended_action": {
    "target_allocations": {
      "aave_usdc": 0.50,
      "curve_3pool": 0.30,
      "uniswap_usdc_dai": 0.15,
      "reserve": 0.05
    },
    "confidence": 0.87,
    "expected_yield_improvement": 0.008,
    "gas_cost_estimate": 45.23
  },
  "should_rebalance": true,
  "reasoning": "AAVE rate trending up, Curve stable, gas optimal"
}
```

### 4.3 Layer 3 → Layer 4 (AI → Execution)

**Protocol:** Message Queue (Redis Pub/Sub) + REST  
**Format:** JSON  
**Authentication:** JWT + HSM signature

**Queue:** `rebalancing_queue`

**Message:**
```json
{
  "transaction_id": "tx_20260208_001",
  "timestamp": "2026-02-08T12:05:00Z",
  "target_allocations": {
    "aave_usdc": 0.50,
    "curve_3pool": 0.30,
    "uniswap_usdc_dai": 0.15,
    "reserve": 0.05
  },
  "current_allocations": {
    "aave_usdc": 0.45,
    "curve_3pool": 0.35,
    "uniswap_usdc_dai": 0.15,
    "reserve": 0.05
  },
  "expected_gas_cost": 45.23,
  "max_slippage": 0.005,
  "deadline": "2026-02-08T12:15:00Z",
  "safety_checks": {
    "all_risk_scores_valid": true,
    "no_kill_switches_active": true,
    "sufficient_gas_balance": true,
    "within_rebalance_cooldown": true
  },
  "signature": "0x..."  // Keeper signature
}
```

## 5. Data Flow: End-to-End Example

### Scenario: Detect opportunity and execute rebalancing

**Step 1: Data Collection (Every 15 seconds)**
```
The Graph → New Aave rate update event detected
Alchemy → Query current TVL via eth_call
Dune → Fetch 30-day volatility (cached, hourly update)
↓
Data Aggregator normalizes and stores in TimescaleDB
Redis cache updated with latest values
```

**Step 2: Risk Assessment (Every 60 seconds)**
```
Risk Engine pulls latest protocol data from Redis
XGBoost model scores each protocol:
  - AAVE USDC: 25/100 (LOW)
  - Curve 3pool: 18/100 (LOW)
  - Uniswap USDC/DAI: 32/100 (MEDIUM)

Kill-Switch checks (all pass):
  ✓ USDC peg: $0.9998
  ✓ TVL stable
  ✓ Utilization: 68%
  ✓ Oracle: Fresh (updated 10s ago)

Output: All protocols safe for allocation
```

**Step 3: AI Inference (Every 5 minutes)**
```
LSTM Yield Predictor:
  - AAVE 7d forecast: 8.2% → 7.8% (declining)
  - Curve 7d forecast: 5.2% → 5.5% (rising)

PPO Agent decides:
  Current: [45% AAVE, 35% Curve, 15% Uni, 5% Reserve]
  Target:  [40% AAVE, 40% Curve, 15% Uni, 5% Reserve]
  Reason: Curve yield rising, reduce AAVE exposure

Gas Optimizer:
  - Current gas: 28 gwei
  - Estimated cost: $45
  - Expected yield improvement: 0.8% = $8,000/year
  - ROI: 178x (worth it!)

Decision: REBALANCE
```

**Step 4: Transaction Building**
```
Transaction Builder constructs multicall:
  1. Withdraw 5% ($50K) from AAVE
  2. Deposit $50K to Curve

Tenderly Simulation:
  ✓ Transaction succeeds
  ✓ Gas: 287,432
  ✓ No reverts
  ✓ Final allocations match target

Approved for execution
```

**Step 5: Signing & Broadcasting**
```
HSM (AWS KMS) signs transaction:
  - Nonce: 1234
  - Gas: 287432 * 1.2 (buffer) = 344,918
  - Max Fee: 35 gwei
  - Priority Fee: 2 gwei

Keeper broadcasts via Flashbots Protect:
  - Private mempool (no front-running)
  - Bundle guaranteed inclusion
  - MEV protection enabled

Transaction submitted: 0xabc...123
```

**Step 6: On-Chain Execution**
```
Block 19,234,567:
  StrategyHub.rebalance() called
  
  Checks:
    ✓ Caller is authorized keeper
    ✓ Last rebalance > 6 hours ago
    ✓ Kill switches inactive
    ✓ Allocations sum to 100%
    
  Execution:
    - Withdraw from Aave: $50K USDC
    - Deposit to Curve: $50K USDC
    
  Event emitted:
    RebalanceExecuted(
      oldAllocations=[45,35,15,5],
      newAllocations=[40,40,15,5],
      gasCost=287432,
      timestamp=1739093100
    )

Transaction confirmed ✓
```

**Step 7: Monitoring & Logging**
```
Prometheus metrics updated:
  - portfolio_value_usd: 1,000,234
  - rebalance_count: +1
  - gas_cost_usd: +45.23

Grafana dashboard updated:
  - New allocation chart rendered
  - APY prediction graph updated

PagerDuty: No alerts (all healthy)

Database logged:
  - INSERT INTO rebalance_history
  - UPDATE portfolio_state
```

## 6. Security Architecture

### 6.1 Key Management

```
┌─────────────────────────────────────────────┐
│           KEY HIERARCHY                     │
│                                             │
│  Cold Storage (Governance)                  │
│  └─ Gnosis Safe 3/5 Multi-Sig              │
│     └─ Parameter changes                    │
│     └─ Emergency shutdown                   │
│     └─ Treasury management                  │
│                                             │
│  Warm Storage (Treasury)                    │
│  └─ Gnosis Safe 2/3 Multi-Sig              │
│     └─ Fund management                      │
│     └─ Strategy updates                     │
│                                             │
│  Hot Storage (Keeper)                       │
│  └─ AWS KMS Managed Key                     │
│     └─ Rebalancing only                     │
│     └─ Rate-limited (4/day)                 │
│     └─ Amount-limited (<10% TVL/tx)         │
└─────────────────────────────────────────────┘
```

### 6.2 Defense in Depth

**Layer 1: Input Validation**
- All external data validated and sanitized
- Price feeds checked for staleness (<1 hour)
- Transaction parameters bounded (slippage, gas)

**Layer 2: Business Logic**
- Risk scores must pass threshold (< 70)
- Minimum time between rebalances (6 hours)
- Maximum position sizes enforced (50%)

**Layer 3: Kill Switches**
- 5 on-chain triggers (automatic)
- 3 off-chain ML anomaly detectors
- Multi-sig can emergency pause

**Layer 4: Monitoring & Alerting**
- Real-time Prometheus metrics
- PagerDuty critical alerts
- Automated incident response

**Layer 5: Audit & Recovery**
- All transactions logged to TimescaleDB
- Immutable event logs on-chain
- Time-travel queries for forensics

## 7. Scalability Considerations

### Current Architecture Limits

| Component | Current Capacity | Bottleneck |
|-----------|------------------|------------|
| Data Ingestion | 100 protocols | The Graph rate limits |
| Risk Scoring | 1000 calculations/min | XGBoost inference CPU |
| ML Inference | 100 predictions/min | LSTM GPU memory |
| Transaction Broadcasting | 10 tx/min | RPC rate limits |

### Scaling Strategies

**Horizontal Scaling:**
- Multiple data ingestion workers (Celery + Redis queue)
- Load-balanced FastAPI ML inference servers
- Read replicas for TimescaleDB

**Vertical Scaling:**
- GPU instances for ML (AWS p3.2xlarge)
- Larger database instances (RDS db.r6g.2xlarge)

**Caching:**
- Redis for hot protocol data (15s TTL)
- CDN for API responses (user dashboards)

**Optimization:**
- ONNX export for faster ML inference
- Quantization (FP16) for models
- Batch processing where possible

## 8. Disaster Recovery

### Backup Strategy

**Database:**
- Automated daily snapshots (RDS)
- Point-in-time recovery (7 days)
- Cross-region replication (optional)

**Models:**
- S3 versioning for trained models
- Git LFS for model checkpoints

**Smart Contracts:**
- Verified source code on Etherscan
- Multi-sig recovery procedures

### Incident Response Time

| Severity | Response Time | Recovery Time |
|----------|---------------|---------------|
| P0 (Funds at risk) | <15 min | <1 hour |
| P1 (Kill switch) | <1 hour | <4 hours |
| P2 (Degraded) | <4 hours | <24 hours |
| P3 (Minor) | <24 hours | <1 week |

---

**Next:** See [API_SPEC.md](./API_SPEC.md) for detailed API documentation.
