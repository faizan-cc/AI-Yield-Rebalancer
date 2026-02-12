#!/bin/bash
# Script to get Aave-compatible USDC on Base Sepolia

echo "=================================================="
echo "Getting USDC for Base Sepolia Testing"
echo "=================================================="
echo ""

AAVE_USDC="0x036CbD53842c5426634e7929541eC2318f3dCF7e"
YOUR_USDC="0xba50Cd2A20f6DA35D788639E581bca8d0B5d4D5f"
WALLET="0x370e3E98173D667939479373B915BBAB3Eaa029F"

echo "Your wallet: $WALLET"
echo "Need USDC at: $AAVE_USDC (Aave-compatible)"
echo "You have USDC at: $YOUR_USDC (Not Aave-compatible, 10,000 balance)"
echo ""

echo "=================================================="
echo "Option 1: Aave Faucet (Recommended)"
echo "=================================================="
echo "1. Visit: https://faucet.aave.com/"
echo "2. Select 'Base Sepolia' network"
echo "3. Connect wallet: $WALLET"
echo "4. Request USDC tokens"
echo ""

echo "=================================================="
echo "Option 2: Base Sepolia Faucets"
echo "=================================================="
echo "1. Coinbase Faucet:"
echo "   https://portal.cdp.coinbase.com/products/faucet"
echo "   - May provide test USDC on Base Sepolia"
echo ""
echo "2. Alchemy Faucet:"
echo "   https://www.alchemy.com/faucets/base-sepolia"
echo "   - Primarily for ETH, check if USDC available"
echo ""
echo "3. QuickNode Faucet:"
echo "   https://faucet.quicknode.com/base/sepolia"
echo ""

echo "=================================================="
echo "Option 3: Use Testnet DEX/Bridge"
echo "=================================================="
echo "If you have ETH on Base Sepolia:"
echo "1. Find a Base Sepolia DEX (Uniswap V3 testnet)"
echo "2. Swap ETH -> USDC (Aave version)"
echo "3. Current ETH balance: 0.0681 ETH"
echo ""

echo "=================================================="
echo "Option 4: Mint Test USDC (If Contract Allows)"
echo "=================================================="
echo "Some test USDC contracts have public mint functions."
echo "Let me check..."
echo ""

# Check if USDC contract has a mint function
python3 << 'EOF'
import sys
sys.path.append('.')
from src.execution.contract_manager import ContractManager
from web3 import Web3

usdc_address = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"
cm = ContractManager('base_sepolia')
w3 = cm.w3

# Try to get contract info
try:
    # Get bytecode to verify contract exists
    code = w3.eth.get_code(usdc_address)
    if code == b'' or code == b'0x':
        print("❌ No contract found at this address")
    else:
        print("✅ Contract exists at", usdc_address)
        
        # Try common ERC20 ABI with mint function
        erc20_abi = [
            {"constant":True,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"type":"function"},
            {"constant":True,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"type":"function"},
            {"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},
            {"constant":True,"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"},
            {"constant":False,"inputs":[{"name":"to","type":"address"},{"name":"amount","type":"uint256"}],"name":"mint","outputs":[],"type":"function"},
            {"constant":True,"inputs":[],"name":"totalSupply","outputs":[{"name":"","type":"uint256"}],"type":"function"}
        ]
        
        usdc = w3.eth.contract(address=usdc_address, abi=erc20_abi)
        
        try:
            name = usdc.functions.name().call()
            symbol = usdc.functions.symbol().call()
            decimals = usdc.functions.decimals().call()
            print(f"   Name: {name}")
            print(f"   Symbol: {symbol}")
            print(f"   Decimals: {decimals}")
            
            balance = usdc.functions.balanceOf(cm.account.address).call()
            print(f"   Your balance: {balance / 10**decimals} {symbol}")
            
            total = usdc.functions.totalSupply().call()
            print(f"   Total Supply: {total / 10**decimals} {symbol}")
            
        except Exception as e:
            print(f"   Could not read token info: {e}")
            
except Exception as e:
    print(f"Error checking contract: {e}")

EOF

echo ""
echo "=================================================="
echo "Next Steps After Getting USDC:"
echo "=================================================="
echo "1. Verify you received Aave USDC:"
echo "   python scripts/check_balances_base.py"
echo ""
echo "2. Add USDC to vault's supported assets:"
echo "   python scripts/add_usdc_to_vault_base.py"
echo ""
echo "3. Deposit USDC:"
echo "   python src/execution/deposit_testnet.py --network base_sepolia --asset USDC --amount 100"
echo ""
echo "4. Watch keeper rebalance between WETH (69.77%) and USDC (1.22%):"
echo "   tail -f logs/keeper.log"
echo ""
echo "=================================================="
