"""
Visualize Backtesting Results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv('backtest_results.csv')
df['date'] = pd.to_datetime(df['date'])

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('DeFi Yield Backtesting Results', fontsize=16, fontweight='bold')

# Plot 1: Portfolio Value Over Time
ax1 = axes[0, 0]
for strategy in df['strategy'].unique():
    strategy_data = df[df['strategy'] == strategy]
    ax1.plot(strategy_data['date'], strategy_data['value'], label=strategy, linewidth=2)

ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
ax1.set_xlabel('Date')
ax1.set_ylabel('Portfolio Value ($)')
ax1.set_title('Portfolio Value Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Returns Distribution
ax2 = axes[0, 1]
for strategy in df['strategy'].unique():
    strategy_data = df[df['strategy'] == strategy]
    returns = strategy_data['value'].pct_change().dropna() * 100
    ax2.hist(returns, bins=30, alpha=0.6, label=strategy)

ax2.set_xlabel('Daily Return (%)')
ax2.set_ylabel('Frequency')
ax2.set_title('Returns Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Cumulative Returns
ax3 = axes[1, 0]
for strategy in df['strategy'].unique():
    strategy_data = df[df['strategy'] == strategy].copy()
    strategy_data['cumulative_return'] = (strategy_data['value'] / 10000 - 1) * 100
    ax3.plot(strategy_data['date'], strategy_data['cumulative_return'], label=strategy, linewidth=2)

ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax3.set_xlabel('Date')
ax3.set_ylabel('Cumulative Return (%)')
ax3.set_title('Cumulative Returns')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Number of Positions
ax4 = axes[1, 1]
for strategy in df['strategy'].unique():
    strategy_data = df[df['strategy'] == strategy]
    ax4.plot(strategy_data['date'], strategy_data['positions'], label=strategy, linewidth=2, marker='o', markersize=3)

ax4.set_xlabel('Date')
ax4.set_ylabel('Number of Positions')
ax4.set_title('Active Positions Over Time')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('backtest_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to backtest_visualization.png")

# Create metrics comparison table
print("\n" + "="*60)
print("STRATEGY COMPARISON")
print("="*60)

for strategy in df['strategy'].unique():
    strategy_data = df[df['strategy'] == strategy].copy()
    
    initial_value = strategy_data.iloc[0]['value']
    final_value = strategy_data.iloc[-1]['value']
    total_return = (final_value - initial_value) / initial_value * 100
    
    strategy_data['daily_return'] = strategy_data['value'].pct_change()
    volatility = strategy_data['daily_return'].std() * np.sqrt(365) * 100
    
    days = (strategy_data['date'].max() - strategy_data['date'].min()).days
    years = days / 365.25
    annualized_return = ((final_value / initial_value) ** (1 / years) - 1) * 100
    
    sharpe = annualized_return / volatility if volatility > 0 else 0
    
    rolling_max = strategy_data['value'].cummax()
    drawdown = (strategy_data['value'] - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()
    
    print(f"\n{strategy}:")
    print(f"  Total Return:      {total_return:>7.2f}%")
    print(f"  Annualized Return: {annualized_return:>7.2f}%")
    print(f"  Volatility:        {volatility:>7.2f}%")
    print(f"  Sharpe Ratio:      {sharpe:>7.2f}")
    print(f"  Max Drawdown:      {max_drawdown:>7.2f}%")
    print(f"  Final Value:       ${final_value:>,.2f}")

print("\n" + "="*60)
