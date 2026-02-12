"""
Get exact revert reason for addPool failure
"""
import os
import sys
from pathlib import Path
from web3 import Web3
from eth_abi import decode
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from src.execution.contract_manager import ContractManager

load_dotenv()

DAI_ADDRESS = "0xFF34B3d4Aee8ddCd6F9AFFFB6Fe49bD371b8a357"

def get_revert_reason():
    """Get the exact revert reason"""
    
    contract_manager = ContractManager('sepolia')
    strategy_manager = contract_manager.contracts['StrategyManager']
    aave_adapter = contract_manager.contracts['AaveAdapter']
    account = contract_manager.account
    w3 = contract_manager.w3
    
    print("Testing addPool transaction to get revert reason...")
    print(f"DAI: {DAI_ADDRESS}")
    print(f"AaveAdapter: {aave_adapter.address}")
    
    # Try to call the function and catch the error
    try:
        # Use eth_call to simulate transaction
        result = w3.eth.call({
            'from': account.address,
            'to': strategy_manager.address,
            'data': strategy_manager.encodeABI(
                fn_name='addPool',
                args=[DAI_ADDRESS, aave_adapter.address, "AaveV3"]
            )
        })
        print(f"✅ Call succeeded (shouldn't happen if tx fails): {result.hex()}")
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Call reverted:")
        print(f"Error: {error_msg}")
        
        # Try to extract revert reason
        if "execution reverted" in error_msg.lower():
            # Try to decode the revert reason
            if "0x" in error_msg:
                # Find hex data in error
                parts = error_msg.split("0x")
                for part in parts[1:]:
                    hex_data = "0x" + part.split('"')[0].split("'")[0].split()[0]
                    if len(hex_data) > 10:
                        try:
                            # Try to decode as string
                            if hex_data.startswith("0x08c379a0"):  # Error(string) selector
                                decoded = decode(['string'], bytes.fromhex(hex_data[10:]))
                                print(f"\n🔍 Decoded revert reason: {decoded[0]}")
                                return
                        except:
                            pass
        
        print(f"\n💡 Could not decode revert reason from error message")
        
    # Also try with debug_traceCall if available
    print("\n" + "=" * 60)
    print("Trying eth_estimateGas to get more info...")
    try:
        gas = w3.eth.estimate_gas({
            'from': account.address,
            'to': strategy_manager.address,
            'data': strategy_manager.encodeABI(
                fn_name='addPool',
                args=[DAI_ADDRESS, aave_adapter.address, "AaveV3"]
            )
        })
        print(f"✅ Estimated gas: {gas}")
    except Exception as e:
        print(f"❌ Estimate gas failed: {e}")

if __name__ == '__main__':
    get_revert_reason()
