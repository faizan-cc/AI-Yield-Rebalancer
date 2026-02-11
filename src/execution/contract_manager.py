"""
Contract Manager - Web3 Integration for Smart Contracts

Handles all interactions with deployed smart contracts on testnets/mainnet.
"""

import os
import json
from typing import Dict, List, Tuple, Optional
from decimal import Decimal
from web3 import Web3
from web3.contract import Contract
from eth_account import Account
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContractManager:
    """Manages smart contract interactions for the DeFi yield system"""
    
    def __init__(self, network: str = "sepolia"):
        """
        Initialize contract manager
        
        Args:
            network: Network to connect to (sepolia, base_sepolia, mainnet, base)
        """
        self.network = network
        self.w3 = self._setup_web3()
        self.account = self._setup_account()
        self.contracts: Dict[str, Contract] = {}
        self._load_contracts()
        
    def _setup_web3(self) -> Web3:
        """Setup Web3 connection"""
        rpc_urls = {
            "sepolia": os.getenv("SEPOLIA_RPC_URL"),
            "base_sepolia": os.getenv("BASE_SEPOLIA_RPC_URL"),
            "mainnet": os.getenv("ETHEREUM_RPC_URL"),
            "base": os.getenv("BASE_RPC_URL")
        }
        
        rpc_url = rpc_urls.get(self.network)
        if not rpc_url:
            raise ValueError(f"No RPC URL configured for {self.network}")
            
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        if not w3.is_connected():
            raise ConnectionError(f"Failed to connect to {self.network}")
            
        logger.info(f"✓ Connected to {self.network}")
        logger.info(f"  Chain ID: {w3.eth.chain_id}")
        logger.info(f"  Latest block: {w3.eth.block_number}")
        
        return w3
        
    def _setup_account(self) -> Account:
        """Setup account from private key"""
        private_key = os.getenv("DEPLOYER_PRIVATE_KEY")
        if not private_key:
            raise ValueError("DEPLOYER_PRIVATE_KEY not set in .env")
            
        account = Account.from_key(private_key)
        balance = self.w3.eth.get_balance(account.address)
        
        logger.info(f"✓ Loaded account: {account.address}")
        logger.info(f"  Balance: {self.w3.from_wei(balance, 'ether')} ETH")
        
        return account
        
    def _load_contracts(self):
        """Load deployed contract addresses and ABIs"""
        deployments_file = f"deployments/{self.network}.json"
        
        if not os.path.exists(deployments_file):
            logger.warning(f"No deployments found for {self.network}")
            return
            
        with open(deployments_file, 'r') as f:
            deployments = json.load(f)
            
        # Load each contract
        for name, deployment in deployments.items():
            address = deployment.get("address")
            abi_file = deployment.get("abi")
            
            if address and abi_file:
                with open(abi_file, 'r') as f:
                    abi = json.load(f)
                    
                contract = self.w3.eth.contract(
                    address=self.w3.to_checksum_address(address),
                    abi=abi
                )
                self.contracts[name] = contract
                logger.info(f"✓ Loaded {name}: {address}")
                
    def deploy_contract(
        self,
        name: str,
        abi_file: str,
        bytecode_file: str,
        constructor_args: List = None
    ) -> str:
        """
        Deploy a new contract
        
        Args:
            name: Contract name
            abi_file: Path to ABI JSON
            bytecode_file: Path to bytecode
            constructor_args: Constructor arguments
            
        Returns:
            Deployed contract address
        """
        # Load ABI and bytecode
        with open(abi_file, 'r') as f:
            abi = json.load(f)
        with open(bytecode_file, 'r') as f:
            bytecode = f.read().strip()
            
        # Create contract instance
        Contract = self.w3.eth.contract(abi=abi, bytecode=bytecode)
        
        # Build constructor transaction
        constructor = Contract.constructor(*(constructor_args or []))
        
        # Estimate gas
        gas_estimate = constructor.estimate_gas({
            'from': self.account.address
        })
        
        # Build transaction
        transaction = constructor.build_transaction({
            'from': self.account.address,
            'gas': int(gas_estimate * 1.2),  # 20% buffer
            'gasPrice': self.w3.eth.gas_price,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'chainId': self.w3.eth.chain_id
        })
        
        # Sign and send
        signed = self.account.sign_transaction(transaction)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        
        logger.info(f"⏳ Deploying {name}...")
        logger.info(f"  TX: {tx_hash.hex()}")
        
        # Wait for receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
        
        if receipt['status'] == 1:
            address = receipt['contractAddress']
            logger.info(f"✅ {name} deployed at {address}")
            logger.info(f"   Gas used: {receipt['gasUsed']}")
            
            # Save deployment
            self._save_deployment(name, address, abi_file)
            
            # Load into contracts dict
            self.contracts[name] = self.w3.eth.contract(
                address=address,
                abi=abi
            )
            
            return address
        else:
            raise Exception(f"Contract deployment failed: {receipt}")
            
    def _save_deployment(self, name: str, address: str, abi_file: str):
        """Save deployment info to file"""
        os.makedirs("deployments", exist_ok=True)
        deployments_file = f"deployments/{self.network}.json"
        
        deployments = {}
        if os.path.exists(deployments_file):
            with open(deployments_file, 'r') as f:
                deployments = json.load(f)
                
        deployments[name] = {
            "address": address,
            "abi": abi_file,
            "deployed_at": self.w3.eth.block_number
        }
        
        with open(deployments_file, 'w') as f:
            json.dump(deployments, f, indent=2)
            
    def call_function(
        self,
        contract_name: str,
        function_name: str,
        *args,
        value: int = 0,
        gas_limit: Optional[int] = None
    ) -> any:
        """
        Call a contract function (state-changing)
        
        Args:
            contract_name: Name of contract
            function_name: Function to call
            *args: Function arguments
            value: ETH value to send (wei)
            gas_limit: Custom gas limit
            
        Returns:
            Transaction receipt
        """
        contract = self.contracts.get(contract_name)
        if not contract:
            raise ValueError(f"Contract {contract_name} not loaded")
            
        # Get function
        func = getattr(contract.functions, function_name)
        
        # Build transaction
        transaction = func(*args).build_transaction({
            'from': self.account.address,
            'value': value,
            'gasPrice': self.w3.eth.gas_price,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'chainId': self.w3.eth.chain_id
        })
        
        # Estimate gas if not provided
        if gas_limit:
            transaction['gas'] = gas_limit
        else:
            transaction['gas'] = self.w3.eth.estimate_gas(transaction)
            
        # Sign and send
        signed = self.account.sign_transaction(transaction)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        
        logger.info(f"⏳ Calling {contract_name}.{function_name}()")
        logger.info(f"  TX: {tx_hash.hex()}")
        
        # Wait for receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
        
        if receipt['status'] == 1:
            logger.info(f"✅ Transaction successful")
            logger.info(f"   Gas used: {receipt['gasUsed']}")
        else:
            logger.error(f"❌ Transaction failed")
            
        return receipt
        
    def read_function(
        self,
        contract_name: str,
        function_name: str,
        *args
    ) -> any:
        """
        Read from contract (view function)
        
        Args:
            contract_name: Name of contract
            function_name: Function to call
            *args: Function arguments
            
        Returns:
            Function return value
        """
        contract = self.contracts.get(contract_name)
        if not contract:
            raise ValueError(f"Contract {contract_name} not loaded")
            
        func = getattr(contract.functions, function_name)
        return func(*args).call()
        
    def get_vault_info(self) -> Dict:
        """Get current vault information"""
        vault = self.contracts.get("YieldVault")
        if not vault:
            return {}
            
        return {
            "tvl": vault.functions.totalValueLocked().call(),
            "total_shares": vault.functions.totalShares().call(),
            "last_rebalance": vault.functions.lastRebalanceTime().call(),
            "is_rebalance_due": vault.functions.isRebalanceDue().call()
        }
        
    def approve_token(
        self,
        token_address: str,
        spender_address: str,
        amount: int
    ) -> any:
        """Approve token spending"""
        # ERC20 ABI (approve function only)
        erc20_abi = [
            {
                "constant": False,
                "inputs": [
                    {"name": "_spender", "type": "address"},
                    {"name": "_value", "type": "uint256"}
                ],
                "name": "approve",
                "outputs": [{"name": "", "type": "bool"}],
                "type": "function"
            }
        ]
        
        token = self.w3.eth.contract(
            address=self.w3.to_checksum_address(token_address),
            abi=erc20_abi
        )
        
        transaction = token.functions.approve(
            self.w3.to_checksum_address(spender_address),
            amount
        ).build_transaction({
            'from': self.account.address,
            'gasPrice': self.w3.eth.gas_price,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'chainId': self.w3.eth.chain_id
        })
        
        signed = self.account.sign_transaction(transaction)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)


if __name__ == "__main__":
    # Example usage
    manager = ContractManager("sepolia")
    
    # Get vault info
    info = manager.get_vault_info()
    print(f"\nVault Info:")
    print(f"  TVL: {info.get('tvl', 0)}")
    print(f"  Total Shares: {info.get('total_shares', 0)}")
    print(f"  Rebalance Due: {info.get('is_rebalance_due', False)}")
