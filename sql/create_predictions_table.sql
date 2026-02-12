-- ML Predictions Table
-- Stores ML-generated predictions for tracking and validation

CREATE TABLE IF NOT EXISTS ml_predictions (
    prediction_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    network VARCHAR(50) NOT NULL,
    pool_address VARCHAR(42) NOT NULL,
    asset_address VARCHAR(42) NOT NULL,
    protocol_name VARCHAR(100),
    
    -- Prediction values
    predicted_apy DECIMAL(10, 4) NOT NULL,
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('low', 'medium', 'high')),
    confidence_score DECIMAL(5, 2) NOT NULL,
    
    -- Actual outcome (filled in later for validation)
    actual_apy DECIMAL(10, 4),
    prediction_error DECIMAL(10, 4),
    
    -- Model metadata
    model_version VARCHAR(50),
    lstm_prediction DECIMAL(10, 4),
    xgboost_risk_score DECIMAL(5, 2),
    
    -- Transaction info
    update_tx_hash VARCHAR(66),
    rebalance_tx_hash VARCHAR(66),
    
    -- Indexing
    CONSTRAINT unique_prediction UNIQUE (timestamp, pool_address, network)
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_predictions_pool ON ml_predictions(pool_address, network, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON ml_predictions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_asset ON ml_predictions(asset_address, network);

-- Rebalancing History Table
-- Tracks all rebalancing events
CREATE TABLE IF NOT EXISTS rebalance_history (
    rebalance_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    network VARCHAR(50) NOT NULL,
    vault_address VARCHAR(42) NOT NULL,
    
    -- Rebalancing details
    asset_address VARCHAR(42) NOT NULL,
    total_assets DECIMAL(30, 18) NOT NULL,
    
    -- Transaction info
    tx_hash VARCHAR(66) NOT NULL,
    gas_used INTEGER NOT NULL,
    gas_price DECIMAL(30, 18) NOT NULL,
    
    -- Success status
    status VARCHAR(20) NOT NULL CHECK (status IN ('pending', 'success', 'failed')),
    error_message TEXT,
    
    CONSTRAINT unique_rebalance UNIQUE (tx_hash)
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_rebalance_vault ON rebalance_history(vault_address, network, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_rebalance_timestamp ON rebalance_history(timestamp DESC);

-- Pool Allocations Table
-- Tracks allocation changes from rebalancing
CREATE TABLE IF NOT EXISTS pool_allocations (
    allocation_id SERIAL PRIMARY KEY,
    rebalance_id INTEGER REFERENCES rebalance_history(rebalance_id) ON DELETE CASCADE,
    
    pool_address VARCHAR(42) NOT NULL,
    protocol_name VARCHAR(100),
    
    -- Allocation percentages
    allocation_percentage INTEGER NOT NULL CHECK (allocation_percentage >= 0 AND allocation_percentage <= 100),
    
    -- Amounts
    allocated_amount DECIMAL(30, 18) NOT NULL,
    
    -- APY at time of allocation
    pool_apy DECIMAL(10, 4),
    predicted_apy DECIMAL(10, 4),
    risk_level VARCHAR(20)
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_allocations_rebalance ON pool_allocations(rebalance_id);
CREATE INDEX IF NOT EXISTS idx_allocations_pool ON pool_allocations(pool_address);

-- Prediction Performance View
-- Track prediction accuracy
CREATE OR REPLACE VIEW prediction_performance AS
SELECT 
    DATE(timestamp) as date,
    pool_address,
    protocol_name,
    COUNT(*) as total_predictions,
    AVG(predicted_apy) as avg_predicted_apy,
    AVG(actual_apy) as avg_actual_apy,
    AVG(ABS(prediction_error)) as avg_absolute_error,
    AVG(confidence_score) as avg_confidence,
    COUNT(CASE WHEN actual_apy IS NOT NULL THEN 1 END) as validated_predictions
FROM ml_predictions
GROUP BY DATE(timestamp), pool_address, protocol_name
ORDER BY date DESC;

-- Rebalancing Summary View
-- Track rebalancing performance
CREATE OR REPLACE VIEW rebalancing_summary AS
SELECT 
    DATE(rh.timestamp) as date,
    rh.network,
    COUNT(rh.rebalance_id) as total_rebalances,
    COUNT(CASE WHEN rh.status = 'success' THEN 1 END) as successful_rebalances,
    AVG(rh.gas_used) as avg_gas_used,
    SUM(rh.gas_used * rh.gas_price) / 1e18 as total_gas_cost_eth,
    AVG(pa.pool_apy) as avg_pool_apy
FROM rebalance_history rh
LEFT JOIN pool_allocations pa ON rh.rebalance_id = pa.rebalance_id
WHERE rh.status = 'success'
GROUP BY DATE(rh.timestamp), rh.network
ORDER BY date DESC;

-- Comments for documentation
COMMENT ON TABLE ml_predictions IS 'Stores ML model predictions for pool APYs and risk levels';
COMMENT ON TABLE rebalance_history IS 'Tracks all rebalancing transactions and their outcomes';
COMMENT ON TABLE pool_allocations IS 'Records how assets were allocated across pools during rebalancing';
COMMENT ON VIEW prediction_performance IS 'Aggregated view of ML prediction accuracy over time';
COMMENT ON VIEW rebalancing_summary IS 'Aggregated view of rebalancing performance and costs';
