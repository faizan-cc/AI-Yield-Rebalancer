#!/bin/bash
# Quick script to check ML predictions in database

DB_NAME="defi_yield_db"

echo "=========================================="
echo "ML PREDICTIONS DASHBOARD"
echo "=========================================="
echo ""

echo "📊 Total Predictions:"
psql -d $DB_NAME -c "SELECT COUNT(*) as total FROM ml_predictions;" -t

echo ""
echo "🕐 Latest 5 Predictions:"
psql -d $DB_NAME -c "
SELECT 
    timestamp::timestamp(0) as time,
    SUBSTRING(pool_address, 1, 10) || '...' as pool,
    predicted_apy || '%' as apy,
    risk_level as risk,
    confidence_score || '%' as confidence
FROM ml_predictions 
ORDER BY timestamp DESC 
LIMIT 5;
" 

echo ""
echo "📈 Average APY by Risk Level:"
psql -d $DB_NAME -c "
SELECT 
    risk_level,
    COUNT(*) as count,
    ROUND(AVG(predicted_apy)::numeric, 2) || '%' as avg_apy,
    ROUND(AVG(confidence_score)::numeric, 2) || '%' as avg_confidence
FROM ml_predictions 
GROUP BY risk_level
ORDER BY risk_level;
"

echo ""
echo "🔄 Rebalancing Events:"
psql -d $DB_NAME -c "
SELECT 
    COUNT(*) as total_rebalances,
    COUNT(CASE WHEN status = 'success' THEN 1 END) as successful,
    ROUND(AVG(gas_used)::numeric, 0) as avg_gas,
    ROUND(SUM(gas_used * gas_price) / 1e18::numeric, 6) || ' ETH' as total_gas_cost
FROM rebalance_history;
" -t

echo ""
echo "=========================================="
echo "Refresh with: ./scripts/check_predictions.sh"
echo "=========================================="
