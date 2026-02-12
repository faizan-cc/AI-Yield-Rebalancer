"""
Testnet Validation Dashboard

Real-time monitoring and validation of testnet performance vs backtest predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="Testnet Validation | DeFi Yield",
    page_icon="🧪",
    layout="wide"
)

# Database connection
@st.cache_resource
def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', 5432),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )

@st.cache_data(ttl=60)
def load_testnet_results():
    """Load testnet rebalancing results"""
    conn = get_db_connection()
    
    query = """
        SELECT 
            strategy_name,
            network,
            timestamp,
            tx_hash,
            gas_cost_eth,
            slippage_actual,
            portfolio_value_before,
            portfolio_value_after,
            status
        FROM testnet_rebalances
        ORDER BY timestamp DESC
        LIMIT 1000
    """
    
    try:
        df = pd.read_sql(query, conn)
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_backtest_results():
    """Load backtest results for comparison"""
    try:
        df = pd.read_csv('backtest_results.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except:
        return pd.DataFrame()

def calculate_metrics(df, strategy_name):
    """Calculate performance metrics"""
    if len(df) == 0:
        return {}
    
    strategy_data = df[df['strategy_name'] == strategy_name].copy()
    
    if len(strategy_data) == 0:
        return {}
    
    # Calculate return
    initial_value = strategy_data.iloc[-1]['portfolio_value_before']
    final_value = strategy_data.iloc[0]['portfolio_value_after']
    total_return = (final_value - initial_value) / initial_value
    
    # Calculate other metrics
    values = strategy_data['portfolio_value_after'].values[::-1]
    daily_returns = np.diff(values) / values[:-1]
    
    metrics = {
        'total_return': total_return,
        'final_value': final_value,
        'rebalances': len(strategy_data),
        'avg_gas_eth': strategy_data['gas_cost_eth'].mean(),
        'max_gas_eth': strategy_data['gas_cost_eth'].max(),
        'avg_slippage': strategy_data['slippage_actual'].mean(),
        'max_slippage': strategy_data['slippage_actual'].max(),
        'success_rate': (strategy_data['status'] == 'success').mean(),
        'volatility': np.std(daily_returns) * np.sqrt(365) if len(daily_returns) > 0 else 0
    }
    
    return metrics

# Header
st.title("🧪 Testnet Validation Dashboard")
st.markdown("Real-time validation of testnet performance vs backtest predictions")

# Load data
testnet_data = load_testnet_results()
backtest_data = load_backtest_results()

if len(testnet_data) == 0:
    st.warning("⚠️ No testnet data available yet. Deploy contracts and run first rebalance.")
    st.info("""
    **To get started:**
    1. Deploy contracts to testnet: `npx hardhat run scripts/deploy.js --network sepolia`
    2. Fund vault with testnet tokens
    3. Execute first rebalance
    4. Data will appear here automatically
    """)
    st.stop()

# Strategy selector
strategies = testnet_data['strategy_name'].unique()
selected_strategy = st.selectbox("Select Strategy", strategies)

# Calculate metrics
testnet_metrics = calculate_metrics(testnet_data, selected_strategy)

# Get backtest comparison
if len(backtest_data) > 0:
    backtest_strategy = backtest_data[backtest_data['strategy'] == selected_strategy]
    if len(backtest_strategy) > 0:
        backtest_return = (backtest_strategy.iloc[-1]['value'] - 10000) / 10000
        backtest_sharpe = 22.0  # From backtest results
        backtest_max_dd = -0.0011
    else:
        backtest_return = 0.079
        backtest_sharpe = 22.0
        backtest_max_dd = -0.0011
else:
    backtest_return = 0.079
    backtest_sharpe = 22.0
    backtest_max_dd = -0.0011

# Key metrics
st.markdown("---")
st.subheader("📊 Performance Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    testnet_return = testnet_metrics.get('total_return', 0)
    deviation = testnet_return - backtest_return
    st.metric(
        "Testnet Return",
        f"{testnet_return:.2%}",
        delta=f"{deviation:.2%} vs backtest"
    )
    
    # Status indicator
    if abs(deviation) < 0.50 * abs(backtest_return):
        st.success("✅ Within ±50% of backtest")
    else:
        st.error("❌ Deviation >50%")

with col2:
    avg_gas = testnet_metrics.get('avg_gas_eth', 0)
    st.metric(
        "Avg Gas Cost",
        f"{avg_gas:.6f} ETH",
        delta="✅" if avg_gas < 0.01 else "❌ Too high"
    )

with col3:
    success_rate = testnet_metrics.get('success_rate', 0)
    st.metric(
        "Success Rate",
        f"{success_rate:.1%}",
        delta="✅" if success_rate > 0.95 else "⚠️ Low"
    )

with col4:
    max_slippage = testnet_metrics.get('max_slippage', 0)
    st.metric(
        "Max Slippage",
        f"{max_slippage:.4%}",
        delta="✅" if max_slippage < 0.005 else "❌ Too high"
    )

# Detailed metrics
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Testnet Metrics")
    metrics_df = pd.DataFrame({
        'Metric': [
            'Total Return',
            'Final Value',
            'Rebalances',
            'Avg Gas (ETH)',
            'Max Gas (ETH)',
            'Avg Slippage',
            'Max Slippage',
            'Volatility'
        ],
        'Value': [
            f"{testnet_metrics.get('total_return', 0):.2%}",
            f"${testnet_metrics.get('final_value', 0):,.2f}",
            f"{testnet_metrics.get('rebalances', 0)}",
            f"{testnet_metrics.get('avg_gas_eth', 0):.6f}",
            f"{testnet_metrics.get('max_gas_eth', 0):.6f}",
            f"{testnet_metrics.get('avg_slippage', 0):.4%}",
            f"{testnet_metrics.get('max_slippage', 0):.4%}",
            f"{testnet_metrics.get('volatility', 0):.2%}"
        ]
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("📈 Backtest Comparison")
    comparison_df = pd.DataFrame({
        'Metric': [
            'Expected Return',
            'Actual Return',
            'Deviation',
            'Expected Sharpe',
            'Expected Max DD',
            'Status'
        ],
        'Value': [
            f"{backtest_return:.2%}",
            f"{testnet_return:.2%}",
            f"{abs(testnet_return - backtest_return):.2%}",
            f"{backtest_sharpe:.2f}",
            f"{backtest_max_dd:.2%}",
            "✅ PASS" if abs(testnet_return - backtest_return) < 0.50 * abs(backtest_return) else "❌ FAIL"
        ]
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Performance chart
st.markdown("---")
st.subheader("📈 Testnet vs Backtest Performance")

strategy_testnet = testnet_data[testnet_data['strategy_name'] == selected_strategy].copy()
strategy_testnet = strategy_testnet.sort_values('timestamp')

fig = go.Figure()

# Testnet performance
fig.add_trace(go.Scatter(
    x=strategy_testnet['timestamp'],
    y=strategy_testnet['portfolio_value_after'],
    name='Testnet (Actual)',
    line=dict(color='green', width=3),
    mode='lines+markers'
))

# Backtest performance (if available)
if len(backtest_strategy) > 0:
    fig.add_trace(go.Scatter(
        x=backtest_strategy['date'],
        y=backtest_strategy['value'],
        name='Backtest (Expected)',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    # Add tolerance band (±50%)
    fig.add_trace(go.Scatter(
        x=backtest_strategy['date'],
        y=backtest_strategy['value'] * 1.5,
        fill=None,
        mode='lines',
        line=dict(color='rgba(200, 200, 200, 0.3)'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=backtest_strategy['date'],
        y=backtest_strategy['value'] * 0.5,
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(200, 200, 200, 0.3)'),
        name='Acceptable Range (±50%)'
    ))

fig.update_layout(
    title="Portfolio Value Over Time",
    xaxis_title="Date",
    yaxis_title="Portfolio Value ($)",
    hovermode='x unified',
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Gas costs over time
st.markdown("---")
st.subheader("⛽ Gas Cost Analysis")

fig_gas = go.Figure()

fig_gas.add_trace(go.Scatter(
    x=strategy_testnet['timestamp'],
    y=strategy_testnet['gas_cost_eth'],
    name='Gas Cost',
    mode='lines+markers',
    line=dict(color='orange')
))

# Add limit line
fig_gas.add_hline(
    y=0.01,
    line_dash="dash",
    line_color="red",
    annotation_text="Target: 0.01 ETH"
)

fig_gas.update_layout(
    title="Gas Cost per Rebalance",
    xaxis_title="Date",
    yaxis_title="Gas Cost (ETH)",
    hovermode='x unified',
    height=400
)

st.plotly_chart(fig_gas, use_container_width=True)

# Slippage analysis
col1, col2 = st.columns(2)

with col1:
    fig_slippage = go.Figure()
    fig_slippage.add_trace(go.Scatter(
        x=strategy_testnet['timestamp'],
        y=strategy_testnet['slippage_actual'] * 100,
        name='Slippage',
        mode='lines+markers',
        line=dict(color='purple')
    ))
    fig_slippage.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="red",
        annotation_text="Limit: 0.5%"
    )
    fig_slippage.update_layout(
        title="Slippage per Rebalance",
        xaxis_title="Date",
        yaxis_title="Slippage (%)",
        height=400
    )
    st.plotly_chart(fig_slippage, use_container_width=True)

with col2:
    # Success rate pie chart
    success_counts = strategy_testnet['status'].value_counts()
    fig_success = go.Figure(data=[go.Pie(
        labels=success_counts.index,
        values=success_counts.values,
        hole=.3,
        marker_colors=['green', 'red']
    )])
    fig_success.update_layout(
        title="Transaction Success Rate",
        height=400
    )
    st.plotly_chart(fig_success, use_container_width=True)

# Multi-strategy comparison
st.markdown("---")
st.subheader("🏆 Multi-Strategy Comparison")

if len(strategies) > 1:
    comparison_data = []
    
    for strategy in strategies:
        metrics = calculate_metrics(testnet_data, strategy)
        comparison_data.append({
            'Strategy': strategy,
            'Return': metrics.get('total_return', 0),
            'Rebalances': metrics.get('rebalances', 0),
            'Avg Gas': metrics.get('avg_gas_eth', 0),
            'Success Rate': metrics.get('success_rate', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Bar chart
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Bar(
        name='Testnet Return',
        x=comparison_df['Strategy'],
        y=comparison_df['Return'] * 100,
        marker_color='green'
    ))
    
    fig_comparison.update_layout(
        title="Strategy Returns Comparison",
        xaxis_title="Strategy",
        yaxis_title="Return (%)",
        height=400
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Detailed table
    st.dataframe(
        comparison_df.style.format({
            'Return': '{:.2%}',
            'Avg Gas': '{:.6f}',
            'Success Rate': '{:.1%}'
        }),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Deploy multiple strategies to see comparison")

# Recent transactions
st.markdown("---")
st.subheader("📜 Recent Transactions")

recent_txs = strategy_testnet.head(10)[['timestamp', 'tx_hash', 'gas_cost_eth', 'slippage_actual', 'portfolio_value_after', 'status']]
recent_txs.columns = ['Timestamp', 'Tx Hash', 'Gas (ETH)', 'Slippage', 'Portfolio Value', 'Status']

st.dataframe(
    recent_txs.style.format({
        'Gas (ETH)': '{:.6f}',
        'Slippage': '{:.4%}',
        'Portfolio Value': '${:,.2f}'
    }).applymap(
        lambda x: 'color: green' if x == 'success' else 'color: red',
        subset=['Status']
    ),
    use_container_width=True,
    hide_index=True
)

# Go/No-Go decision
st.markdown("---")
st.subheader("🚦 Go/No-Go Decision Matrix")

col1, col2, col3 = st.columns(3)

checks = {
    'Return Deviation < 50%': abs(testnet_return - backtest_return) < 0.50 * abs(backtest_return),
    'Avg Gas < 0.01 ETH': testnet_metrics.get('avg_gas_eth', 1) < 0.01,
    'Max Slippage < 0.5%': testnet_metrics.get('max_slippage', 1) < 0.005,
    'Success Rate > 95%': testnet_metrics.get('success_rate', 0) > 0.95,
    'Rebalances > 20': testnet_metrics.get('rebalances', 0) > 20
}

passed = sum(checks.values())
total = len(checks)

with col1:
    st.metric("Checks Passed", f"{passed} / {total}")

with col2:
    score = (passed / total) * 10
    st.metric("Score", f"{score:.1f} / 10")

with col3:
    if score >= 8.0:
        st.success("✅ APPROVED")
        st.balloons()
    elif score >= 6.0:
        st.warning("⚠️ CONDITIONAL")
    else:
        st.error("❌ REJECTED")

# Detailed checks
st.markdown("**Validation Checks:**")
for check, passed in checks.items():
    status = "✅" if passed else "❌"
    st.write(f"{status} {check}")

# Recommendation
st.markdown("---")
if score >= 8.0:
    st.success("""
    ### ✅ Recommendation: APPROVED for Mainnet
    
    All critical validation checks passed. Proceed with:
    1. Professional security audit ($15K-25K)
    2. Mainnet deployment with $1K initial capital
    3. 2-week soft launch period
    4. Scale to $10K+ if no issues
    """)
elif score >= 6.0:
    st.warning("""
    ### ⚠️ Recommendation: CONDITIONAL APPROVAL
    
    Most checks passed, but some issues need attention:
    - Review failed checks above
    - Implement fixes in 1-week sprint
    - Re-test affected components
    - Re-run go/no-go decision
    """)
else:
    st.error("""
    ### ❌ Recommendation: REJECTED
    
    Multiple critical failures detected:
    - Conduct post-mortem analysis
    - Identify root causes
    - Implement comprehensive fixes (2-4 weeks)
    - Restart testnet testing from Phase 3
    """)

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refreshes every 60 seconds")
