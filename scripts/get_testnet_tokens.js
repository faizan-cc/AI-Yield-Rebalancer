const hre = require("hardhat");
const { ethers } = require("hardhat");

async function main() {
  const [signer] = await ethers.getSigners();
  console.log("Using account:", signer.address);
  
  const balance = await ethers.provider.getBalance(signer.address);
  console.log("ETH Balance:", ethers.formatEther(balance), "ETH\n");

  // Sepolia token addresses
  const WETH_ADDRESS = "0xC558DBdd856501FCd9aaF1E62eae57A9F0629a3c"; // Aave Sepolia WETH
  const DAI_ADDRESS = "0xFF34B3d4Aee8ddCd6F9AFFFB6Fe49bD371b8a357";  // Aave Sepolia DAI
  const USDC_ADDRESS = "0x94a9D9AC8a22534E3FaCa9F4e7F2E2cf85d5E4C8"; // Aave Sepolia USDC

  console.log("📋 Testnet Token Addresses:");
  console.log("USDC:", USDC_ADDRESS);
  console.log("DAI:", DAI_ADDRESS);
  console.log("WETH:", WETH_ADDRESS);
  console.log("");

  // Option 1: Wrap ETH to WETH
  console.log("═══════════════════════════════════════");
  console.log("OPTION 1: Wrap ETH → WETH");
  console.log("═══════════════════════════════════════\n");
  
  const weth = await ethers.getContractAt(
    ["function deposit() payable", "function balanceOf(address) view returns (uint256)"],
    WETH_ADDRESS
  );

  try {
    const wethBalance = await weth.balanceOf(signer.address);
    console.log("Current WETH balance:", ethers.formatEther(wethBalance), "WETH");
    
    console.log("\n💡 To get WETH, wrap your Sepolia ETH:");
    console.log("   Run: node scripts/wrap_eth.js");
  } catch (error) {
    console.log("⚠️  Could not check WETH balance");
  }

  // Option 2: DAI Faucet Info
  console.log("\n═══════════════════════════════════════");
  console.log("OPTION 2: Get DAI from Faucet");
  console.log("═══════════════════════════════════════\n");
  
  console.log("🚰 DAI Faucet Options:");
  console.log("1. Aave Sepolia Faucet: https://staging.aave.com/faucet/");
  console.log("   - Connect wallet");
  console.log("   - Select Sepolia network");
  console.log("   - Request DAI (may be limited)\n");
  
  console.log("2. Compound Faucet: https://app.compound.finance/");
  console.log("   - Switch to Sepolia testnet");
  console.log("   - Navigate to Faucet section\n");
  
  console.log("3. QuickNode Faucet: https://faucet.quicknode.com/drip");
  console.log("   - Select Sepolia + DAI");
  console.log("   - Enter your address\n");

  console.log("4. Mint directly from DAI test contract:");
  console.log("   - Visit: https://sepolia.etherscan.io/address/" + DAI_ADDRESS + "#writeContract");
  console.log("   - Connect wallet");
  console.log("   - Look for 'mint' or 'faucet' function");
  console.log("   - Try minting some DAI\n");

  // Option 3: Alternative - Just use USDC for now
  console.log("═══════════════════════════════════════");
  console.log("OPTION 3: Test with USDC Only (Recommended)");
  console.log("═══════════════════════════════════════\n");
  
  console.log("✅ You already have USDC!");
  console.log("   You can start testing with just USDC:");
  console.log("   1. Deposit USDC into vault");
  console.log("   2. Test withdraw");
  console.log("   3. Test rebalancing");
  console.log("   4. Add other tokens later\n");

  // Check current token balances
  console.log("═══════════════════════════════════════");
  console.log("YOUR CURRENT BALANCES");
  console.log("═══════════════════════════════════════\n");

  const usdc = await ethers.getContractAt(
    ["function balanceOf(address) view returns (uint256)", "function decimals() view returns (uint8)"],
    USDC_ADDRESS
  );

  try {
    const usdcBalance = await usdc.balanceOf(signer.address);
    const decimals = await usdc.decimals();
    console.log("USDC:", ethers.formatUnits(usdcBalance, decimals), "USDC");
  } catch (error) {
    console.log("USDC: Unable to check (may not have any yet)");
  }

  try {
    const dai = await ethers.getContractAt(
      ["function balanceOf(address) view returns (uint256)"],
      DAI_ADDRESS
    );
    const daiBalance = await dai.balanceOf(signer.address);
    console.log("DAI:", ethers.formatEther(daiBalance), "DAI");
  } catch (error) {
    console.log("DAI: 0 DAI (not obtained yet)");
  }

  try {
    const wethBalance = await weth.balanceOf(signer.address);
    console.log("WETH:", ethers.formatEther(wethBalance), "WETH");
  } catch (error) {
    console.log("WETH: 0 WETH (not wrapped yet)");
  }

  console.log("\n════════════════════════════════════════");
  console.log("RECOMMENDED NEXT STEP");
  console.log("════════════════════════════════════════\n");
  console.log("🎯 Start testing with USDC first!");
  console.log("   Run: node scripts/create_deposit_script.js");
  console.log("   Then: python3 src/execution/deposit_testnet.py\n");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
