"""
Get DAI on Sepolia through multiple methods
"""
import os
import sys
from pathlib import Path
from web3 import Web3
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# Known DAI contracts on Sepolia (we'll try multiple)
DAI_CONTRACTS = [
    {
        'name': 'Aave DAI',
        'address': '0xFF34B3d4Aee8ddCd6F9AFFFB6Fe49bD371b8a357',
        'has_mint': True
    },
    {
        'name': 'MakerDAO DAI',
        'address': '0x68194a729C2450ad26072b3D33ADaCbcef39D574',
        'has_mint': True
    },
    {
        'name': 'Circle DAI',
        'address': '0x3e622317f8C93f7328350cF0B56d9eD4C620C5d6',
        'has_mint': True
    }
]

# ERC20 ABI with mint function
DAI_ABI = [
    {
        "constant": False,
        "inputs": [{"name": "usr", "type": "address"}, {"name": "wad", "type": "uint256"}],
        "name": "mint",
        "outputs": [],
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [{"name": "wad", "type": "uint256"}],
        "name": "mint",
        "outputs": [],
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [],
        "name": "allocateTo",
        "outputs": [],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    }
]

def try_mint_dai(w3, account, contract_info):
    """Try to mint DAI from a specific contract"""
    
    try:
        dai_contract = w3.eth.contract(
            address=contract_info['address'],
            abi=DAI_ABI
        )
        
        print(f"\n🎯 Trying {contract_info['name']} ({contract_info['address']})...")
        
        # Check current balance
        try:
            balance = dai_contract.functions.balanceOf(account.address).call()
            balance_formatted = balance / (10 ** 18)
            print(f"   Current balance: {balance_formatted} DAI")
        except:
            print(f"   ⚠️  Could not check balance")
        
        # Try mint with address and amount
        amount = w3.to_wei(10000, 'ether')  # 10,000 DAI
        
        try:
            print(f"   Attempting mint(address, amount)...")
            
            # Get gas price and nonce
            gas_price = w3.eth.gas_price
            gas_price_buffered = int(gas_price * 1.2)
            nonce = w3.eth.get_transaction_count(account.address, 'pending')
            
            # Try to build transaction
            transaction = dai_contract.functions.mint(
                account.address,
                amount
            ).build_transaction({
                'from': account.address,
                'gas': 200000,
                'gasPrice': gas_price_buffered,
                'nonce': nonce,
                'chainId': w3.eth.chain_id
            })
            
            # Sign and send
            signed_txn = w3.eth.account.sign_transaction(transaction, account.key)
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            print(f"   📤 Transaction sent: {tx_hash.hex()}")
            print(f"   ⏳ Waiting for confirmation...")
            
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
            
            if receipt['status'] == 1:
                print(f"   ✅ Successfully minted 10,000 DAI!")
                print(f"   ⛽ Gas used: {receipt['gasUsed']}")
                
                # Check new balance
                new_balance = dai_contract.functions.balanceOf(account.address).call()
                new_balance_formatted = new_balance / (10 ** 18)
                print(f"   💎 New balance: {new_balance_formatted} DAI")
                
                return True, contract_info['address']
            else:
                print(f"   ❌ Transaction failed")
                return False, None
                
        except Exception as e:
            error_msg = str(e)
            if "execution reverted" in error_msg.lower():
                print(f"   ❌ Mint not allowed (contract reverted)")
            elif "method not found" in error_msg.lower():
                print(f"   ❌ Mint function not available")
            else:
                print(f"   ❌ Error: {error_msg[:100]}")
            
            # Try simple mint without address parameter
            try:
                print(f"   Attempting mint(amount) without address...")
                transaction = dai_contract.functions.mint(amount).build_transaction({
                    'from': account.address,
                    'gas': 200000,
                    'gasPrice': gas_price_buffered,
                    'nonce': nonce,
                    'chainId': w3.eth.chain_id
                })
                
                signed_txn = w3.eth.account.sign_transaction(transaction, account.key)
                tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
                
                print(f"   📤 Transaction sent: {tx_hash.hex()}")
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
                
                if receipt['status'] == 1:
                    print(f"   ✅ Successfully minted DAI!")
                    new_balance = dai_contract.functions.balanceOf(account.address).call()
                    print(f"   💎 New balance: {new_balance / (10**18)} DAI")
                    return True, contract_info['address']
                    
            except Exception as e2:
                print(f"   ❌ Also failed: {str(e2)[:100]}")
            
            return False, None
            
    except Exception as e:
        print(f"   ❌ Contract error: {str(e)[:100]}")
        return False, None

def deploy_mock_dai(w3, account):
    """Deploy a simple mock DAI contract for testing"""
    
    print("\n🏗️  Deploying Mock DAI Contract...")
    
    # Simple mock ERC20 bytecode with public mint
    # This is a minimal ERC20 with mint function
    mock_dai_bytecode = "0x608060405234801561001057600080fd5b506040518060400160405280600981526020017f4d6f636b2044414900000000000000000000000000000000000000000000000081525060009081610055919061029d565b506040518060400160405280600381526020017f44414900000000000000000000000000000000000000000000000000000000008152506001908161009a919061029d565b506012600260006101000a81548160ff021916908360ff16021790555061036f565b600081519050919050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052604160045260246000fd5b7f4e487b7100000000000000000000000000000000000000000000000000000000600052602260045260246000fd5b6000600282049050600182168061014557607f821691505b602082108103610158576101576100fe565b5b50919050565b60008190508160005260206000209050919050565b60006020601f8301049050919050565b600082821b905092915050565b6000600883026101c07fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff82610183565b6101ca8683610183565b95508019841693508086168417925050509392505050565b6000819050919050565b6000819050919050565b600061021161020c610207846101e2565b6101ec565b6101e2565b9050919050565b6000819050919050565b61022b836101f6565b61023f61023782610218565b848454610190565b825550505050565b600090565b610254610247565b61025f818484610222565b505050565b5b818110156102835761027860008261024c565b600181019050610265565b5050565b601f8211156102c85761029981610165565b6102a28461017a565b810160208510156102b1578190505b6102c56102bd8561017a565b830182610264565b50505b505050565b600082821c905092915050565b60006102eb600019846008026102cd565b1980831691505092915050565b600061030483836102da565b9150826002028217905092915050565b61031d826100c4565b67ffffffffffffffff811115610336576103356100cf565b5b610340825461012d565b61034b828285610287565b600060209050601f83116001811461037e576000841561036c578287015190505b61037685826102f8565b8655506103de565b601f19841661038c8661015e565b60005b828110156103b45784890151825560018201915060208501945060208101905061038f565b868310156103d157848901516103cd601f8916826102da565b8355505b6001600288020188555050505b505050505050565b610c52806103ee6000396000f3fe608060405234801561001057600080fd5b50600436106100935760003560e01c8063313ce56711610066578063313ce567146101355780633950935114610153578063395093511461018357806340c10f19146101b357806370a08231146101cf57610093565b806306fdde0314610098578063095ea7b3146100b657806318160ddd146100e657806323b872dd14610104575b600080fd5b6100a06101ff565b6040516100ad9190610834565b60405180910390f35b6100d060048036038101906100cb9190610900565b61028d565b6040516100dd919061095b565b60405180910390f35b6100ee6102b0565b6040516100fb9190610985565b60405180910390f35b61011e600480360381019061011991906109a0565b6102b6565b60405161012c92919061095b565b60405180910390f35b61013d6103a9565b60405161014a9190610a0f565b60405180910390f35b61016d60048036038101906101689190610900565b6103bc565b60405161017a919061095b565b60405180910390f35b61019d60048036038101906101989190610900565b610467565b6040516101aa919061095b565b60405180910390f35b6101cd60048036038101906101c89190610900565b61051a565b005b6101e960048036038101906101e49190610a2a565b6105c8565b6040516101f69190610985565b60405180910390f35b6000805461020c90610a86565b80601f016020809104026020016040519081016040528092919081815260200182805461023890610a86565b80156102855780601f1061025a57610100808354040283529160200191610285565b820191906000526020600020905b81548152906001019060200180831161026857829003601f168201915b505050505081565b600080339050610297818585610610565b610"
    
    print("   ⚠️  Mock DAI deployment requires gas and complex bytecode")
    print("   ❌ Skipping for now - will use alternative method")
    return None

def main():
    """Try to get DAI through various methods"""
    
    # Setup Web3
    rpc_url = os.getenv('SEPOLIA_RPC_URL')
    if not rpc_url:
        print("❌ SEPOLIA_RPC_URL not found in .env")
        return
    
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print("❌ Failed to connect to Sepolia")
        return
    
    print("=" * 60)
    print("GET DAI ON SEPOLIA - MULTIPLE METHODS")
    print("=" * 60)
    print(f"✅ Connected to Sepolia (Chain ID: {w3.eth.chain_id})")
    
    # Get account
    private_key = os.getenv('DEPLOYER_PRIVATE_KEY') or os.getenv('PRIVATE_KEY')
    if not private_key:
        print("❌ DEPLOYER_PRIVATE_KEY or PRIVATE_KEY not found in .env")
        return
    
    account = w3.eth.account.from_key(private_key)
    print(f"📍 Wallet: {account.address}")
    
    # Try each DAI contract
    for contract_info in DAI_CONTRACTS:
        success, dai_address = try_mint_dai(w3, account, contract_info)
        if success:
            print(f"\n✅ SUCCESS! Use this DAI address: {dai_address}")
            return
    
    # If all failed, provide alternatives
    print("\n" + "=" * 60)
    print("❌ Could not mint DAI from any contract")
    print("=" * 60)
    print("\n📋 ALTERNATIVE OPTIONS:")
    print("\n1. SWAP USDC FOR DAI ON UNISWAP:")
    print("   You have 30 USDC. We can swap some for DAI on Uniswap Sepolia")
    print("   Run: python scripts/swap_usdc_to_dai.py --amount 100")
    
    print("\n2. USE USDC ONLY:")
    print("   Continue with just USDC + WETH for multi-asset testing")
    print("   You have: 30 USDC + 0.07112 WETH")
    
    print("\n3. TRY SEPOLIA FAUCETS:")
    print("   - https://sepoliafaucet.com/")
    print("   - https://www.alchemy.com/faucets/ethereum-sepolia")
    print("   - Request DAI if available")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
