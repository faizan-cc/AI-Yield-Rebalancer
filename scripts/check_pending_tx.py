#!/usr/bin/env python3
"""
Check for pending transactions
"""

from web3 import Web3
from dotenv import load_dotenv
import os

load_dotenv()
w3 = Web3(Web3.HTTPProvider(os.getenv('SEPOLIA_RPC_URL')))
address = '0x370e3E98173D667939479373B915BBAB3Eaa029F'

confirmed_nonce = w3.eth.get_transaction_count(address, 'latest')
pending_nonce = w3.eth.get_transaction_count(address, 'pending')

print(f'Confirmed nonce: {confirmed_nonce}')
print(f'Pending nonce: {pending_nonce}')
print(f'Pending transactions: {pending_nonce - confirmed_nonce}')

if pending_nonce > confirmed_nonce:
    print(f'\n⚠️ {pending_nonce - confirmed_nonce} transaction(s) still pending')
    print('Wait 1-2 minutes for them to confirm before restarting keeper')
else:
    print('\n✅ No pending transactions - safe to restart keeper')
