const hre = require("hardhat");
const { ethers } = require("hardhat");
const fs = require("fs");

async function main() {
  const [signer] = await ethers.getSigners();
  console.log("Using account:", signer.address);
  
  // Load deployment
  const deployment = JSON.parse(
    fs.readFileSync("deployments/sepolia_deployment.json", "utf8")
  );
  
  const STRATEGY_MANAGER = deployment.contracts.StrategyManager;
  const AAVE_ADAPTER = deployment.contracts.AaveAdapter;
  
  // Faucet USDC address
  const USDC_ADDRESS = "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238";
  
  console.log("\n📋 Adding Faucet USDC to Strategy Manager...\n");
  console.log("USDC Address:", USDC_ADDRESS);
  console.log("Strategy Manager:", STRATEGY_MANAGER);
  console.log("Aave Adapter:", AAVE_ADAPTER);
  
  const strategyManager = await ethers.getContractAt("StrategyManager", STRATEGY_MANAGER);
  
  // Check if pool already exists
  try {
    const pools = await strategyManager.getActivePools();
    console.log("\nCurrent active pools:", pools.length);
    
    for (let i = 0; i < pools.length; i++) {
      const pool = await strategyManager.pools(i);
      console.log(`Pool ${i}: ${pool.tokenAddress}`);
    }
  } catch (error) {
    console.log("Could not list pools");
  }
  
  // Add the faucet USDC pool
  console.log("\n🔄 Adding faucet USDC/Aave pool...");
  
  try {
    const tx = await strategyManager.addPool(
      USDC_ADDRESS,
      AAVE_ADAPTER,
      "Aave V3 Sepolia USDC"
    );
    
    console.log("Transaction hash:", tx.hash);
    console.log("Waiting for confirmation...");
    
    await tx.wait();
    console.log("✅ Pool added successfully!");
    
    // Get the new pool details
    const pools = await strategyManager.getActivePools();
    console.log("\nTotal active pools now:", pools.length);
    
  } catch (error) {
    if (error.message.includes("Pool already exists")) {
      console.log("✅ Pool already exists - no action needed");
    } else {
      console.error("❌ Error adding pool:", error.message);
      throw error;
    }
  }
  
  // Register aToken for gas optimization
  console.log("\n🔄 Registering aToken...");
  
  const aaveAdapter = await ethers.getContractAt("AaveAdapter", AAVE_ADAPTER);
  
  try {
    const tx = await aaveAdapter.registerAToken(USDC_ADDRESS);
    console.log("Transaction hash:", tx.hash);
    await tx.wait();
    console.log("✅ aToken registered!");
  } catch (error) {
    if (error.message.includes("already registered")) {
      console.log("✅ aToken already registered");
    } else {
      console.error("Note: Could not register aToken:", error.message);
    }
  }
  
  console.log("\n✅ Setup complete! You can now deposit USDC.");
  console.log("Run: python3 src/execution/deposit_testnet.py\n");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
