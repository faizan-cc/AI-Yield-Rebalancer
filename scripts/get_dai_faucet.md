# Getting DAI on Sepolia Testnet

## Method 1: Aave Faucet (Recommended - Easiest)

1. **Visit the Aave Faucet:**
   - URL: https://staging.aave.com/faucet/
   
2. **Connect Your Wallet:**
   - Click "Connect Wallet"
   - Select MetaMask (or your wallet)
   - Make sure you're on Sepolia testnet
   
3. **Get DAI:**
   - Select "DAI" from the token dropdown
   - Click "Faucet" button
   - Approve the transaction in your wallet
   - You'll receive 10,000 DAI tokens
   
4. **DAI Address on Sepolia:**
   ```
   0xFF34B3d4Aee8ddCd6F9AFFFB6Fe49bD371b8a357
   ```

## Method 2: Chainlink Faucet

1. **Visit Chainlink Faucet:**
   - URL: https://faucets.chain.link/sepolia
   
2. **Connect wallet and request testnet tokens**

## Method 3: Direct Contract Interaction (If faucet fails)

Some test DAI contracts have a public `mint()` function. You can try:

```bash
# Using cast (Foundry)
cast send 0xFF34B3d4Aee8ddCd6F9AFFFB6Fe49bD371b8a357 \
  "mint(address,uint256)" \
  YOUR_WALLET_ADDRESS \
  10000000000000000000000 \
  --private-key YOUR_PRIVATE_KEY \
  --rpc-url $SEPOLIA_RPC_URL
```

Or check if there's a mintable test DAI contract on Sepolia.

## Verify Your Balance

After getting DAI, run:

```bash
python scripts/check_balances.py
```

You should see your DAI balance displayed.

## Next Steps

Once you have WETH and DAI:

1. **Add pools to StrategyManager:**
   ```bash
   node scripts/add_pool.js --asset WETH --adapter AaveAdapter
   node scripts/add_pool.js --asset DAI --adapter AaveAdapter
   ```

2. **Deposit tokens into vault:**
   ```bash
   python src/execution/deposit_testnet.py --asset DAI --amount 100
   python src/execution/deposit_testnet.py --asset WETH --amount 0.1
   ```

3. **Update keeper to monitor all 3 pools** (USDC, DAI, WETH)

4. **Restart keeper and watch multi-asset rebalancing!**
