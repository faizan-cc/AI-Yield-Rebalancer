import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Performance data for all versions
versions = {
    'Initial ML\nv1.0': {
        'return': 1.09,
        'sharpe': 5.20,
        'tx_costs': 130,
        'color': '#ff6b6b'
    },
    'Baseline\n(Highest APY)': {
        'return': 1.32,
        'sharpe': 6.26,
        'tx_costs': 130,
        'color': '#95a5a6'
    },
    'Optimized ML\nv2.0': {
        'return': 1.84,
        'sharpe': 7.41,
        'tx_costs': 10,
        'color': '#4ecdc4'
    },
    'Highest APY\n(Current)': {
        'return': 2.09,
        'sharpe': 14.20,
        'tx_costs': 130,
        'color': '#95a5a6'
    },
    'Enhanced ML\nv3.0 ⭐': {
        'return': 2.82,
        'sharpe': 15.21,
        'tx_costs': 3,
        'color': '#2ecc71'
    }
}

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('ML Strategy Evolution - Complete Performance Analysis', 
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Returns comparison
ax1 = axes[0, 0]
names = list(versions.keys())
returns = [v['return'] for v in versions.values()]
colors = [v['color'] for v in versions.values()]

bars = ax1.bar(names, returns, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Total Return (%)', fontsize=12, fontweight='bold')
ax1.set_title('Returns: +159% Improvement (v1.0 → v3.0)', fontsize=13, fontweight='bold')
ax1.axhline(y=2.82, color='green', linestyle='--', alpha=0.3, label='Target: 2.82%')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Plot 2: Sharpe Ratio comparison
ax2 = axes[0, 1]
sharpes = [v['sharpe'] for v in versions.values()]

bars = ax2.bar(names, sharpes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
ax2.set_title('Risk-Adjusted Returns: +192% Improvement', fontsize=13, fontweight='bold')
ax2.axhline(y=15.21, color='green', linestyle='--', alpha=0.3)
ax2.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Plot 3: Transaction Costs
ax3 = axes[1, 0]
tx_costs = [v['tx_costs'] for v in versions.values()]

bars = ax3.bar(names, tx_costs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Transaction Costs ($)', fontsize=12, fontweight='bold')
ax3.set_title('Cost Reduction: -97% (v1.0 → v3.0)', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'${height:.0f}',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Plot 4: Net Profit (Return - TX Costs as %)
ax4 = axes[1, 1]
net_profits = [v['return'] - (v['tx_costs'] / 100) for v in versions.values()]

bars = ax4.bar(names, net_profits, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Net Profit (%)', fontsize=12, fontweight='bold')
ax4.set_title('Net After TX Costs: v3.0 is Clear Winner', fontsize=13, fontweight='bold')
ax4.axhline(y=2.79, color='green', linestyle='--', alpha=0.3)
ax4.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Rotate x-axis labels for better readability
for ax in axes.flat:
    ax.tick_params(axis='x', rotation=15)
    ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('ml_strategy_evolution.png', dpi=300, bbox_inches='tight')
print('✓ Saved evolution comparison to ml_strategy_evolution.png')

# Print summary
print('\n' + '='*60)
print('OPTIMIZATION SUMMARY')
print('='*60)
print(f'\nInitial ML v1.0:      +1.09% return, Sharpe 5.20, TX costs $130')
print(f'Optimized ML v2.0:    +1.84% return, Sharpe 7.41, TX costs $10')
print(f'Enhanced ML v3.0:     +2.82% return, Sharpe 15.21, TX costs $3')
print(f'\nTotal Improvement:    +159% return, +192% Sharpe, -97% costs')
print(f'\nProfit on $10K:       $282 (vs $109 initially) = +$173 extra')
print(f'Profit on $100K:      $2,820 (vs $1,090 initially) = +$1,730 extra')
print('='*60)
