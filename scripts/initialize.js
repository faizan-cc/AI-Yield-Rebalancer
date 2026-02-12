const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

/**
 * Initialize testnet deployment with test pools
 * Run after deploy.js to setup initial pools
 */

async function main() {
  const network = hre.network.name;
  
  console.log(`\n🔧 Initializing testnet on ${network}...\n`);

  // Load deployment
  const deploymentPath = path.join(__dirname, `../deployments/${network}_deployment.json`);
  if (!fs.existsSync(deploymentPath)) {
    console.error("❌ Deploy contracts first (npm run deploy:sepolia)");
    process.exit(1);
  }

  const deployment = JSON.parse(fs.readFileSync(deploymentPath, "utf8"));
  const [deployer] = await hre.ethers.getSigners();

  // Get contract instances
  const strategyManager = await hre.ethers.getContractAt("StrategyManager", deployment.contracts.StrategyManager);

  // Testnet token addresses
  const TOKENS = {
    sepolia: {
      USDC: "0x94a9D9AC8a22534E3FaCa9F4e7F2E2cf85d5E4C8", // Aave Sepolia USDC
      DAI: "0xFF34B3d4Aee8ddCd6F9AFFFB6Fe49bD371b8a357",  // Aave Sepolia DAI
      WETH: "0xC558DBdd856501FCd9aaF1E62eae57A9F0629a3c"  // Aave Sepolia WETH
    },
    base_sepolia: {
      USDC: "0x036CbD53842c5426634e7929541eC2318f3dCF7e", // Base Sepolia USDC
      WETH: "0x4200000000000000000000000000000000000006"  // Base Sepolia WETH
    }
  };

  const tokens = TOKENS[network];
  if (!tokens) {
    console.error(`❌ Network ${network} not configured`);
    process.exit(1);
  }

  console.log("📋 Adding test pools to StrategyManager...\n");

  // Add USDC/Aave pool
  console.log("Adding USDC/Aave pool...");
  try {
    const tx1 = await strategyManager.addPool(
      tokens.USDC,
      deployment.contracts.AaveAdapter,
      "aave_v3"
    );
    await tx1.wait();
    console.log("✅ USDC/Aave pool added");
  } catch (error) {
    console.log("⚠️ USDC/Aave pool might already exist:", error.message);
  }

  // Add DAI/Aave pool (Sepolia only)
  if (network === "sepolia" && tokens.DAI) {
    console.log("Adding DAI/Aave pool...");
    try {
      const tx2 = await strategyManager.addPool(
        tokens.DAI,
        deployment.contracts.AaveAdapter,
        "aave_v3"
      );
      await tx2.wait();
      console.log("✅ DAI/Aave pool added");
    } catch (error) {
      console.log("⚠️ DAI/Aave pool might already exist:", error.message);
    }
  }

  // Add WETH/Aave pool
  console.log("Adding WETH/Aave pool...");
  try {
    const tx3 = await strategyManager.addPool(
      tokens.WETH,
      deployment.contracts.AaveAdapter,
      "aave_v3"
    );
    await tx3.wait();
    console.log("✅ WETH/Aave pool added");
  } catch (error) {
    console.log("⚠️ WETH/Aave pool might already exist:", error.message);
  }

  // Register aTokens for gas efficiency
  console.log("\n🏷️ Registering aTokens...");
  const aaveAdapter = await hre.ethers.getContractAt("AaveAdapter", deployment.contracts.AaveAdapter);
  
  for (const [name, address] of Object.entries(tokens)) {
    try {
      console.log(`Registering ${name}...`);
      const tx = await aaveAdapter.registerAToken(address);
      await tx.wait();
      console.log(`✅ ${name} aToken registered`);
    } catch (error) {
      console.log(`⚠️ ${name} registration failed:`, error.message);
    }
  }

  // Get active pools
  console.log("\n📊 Active pools:");
  const activePools = await strategyManager.getActivePools();
  console.log(`Total pools: ${activePools.length}\n`);

  for (let i = 0; i < activePools.length; i++) {
    const pool = await strategyManager.getPool(activePools[i]);
    console.log(`Pool ${i + 1}:`);
    console.log(`  Token: ${pool.token}`);
    console.log(`  Protocol: ${pool.protocolName}`);
    console.log(`  Active: ${pool.isActive}`);
    console.log(`  APY: ${Number(pool.currentAPY) / 100}%`);
    console.log("");
  }

  console.log("=".repeat(60));
  console.log("✅ Initialization complete!");
  console.log("=".repeat(60));
  
  console.log("\n📋 Next steps:");
  console.log("1. Get testnet tokens from faucets:");
  console.log("   - Aave Sepolia faucet: https://staging.aave.com/faucet/");
  console.log("   - Base Sepolia faucet: https://www.coinbase.com/faucets");
  console.log("\n2. Update pool APYs:");
  console.log("   python src/execution/update_pools.py");
  console.log("\n3. Deposit to vault:");
  console.log("   python src/execution/deposit.py --amount 100");
  console.log("\n4. Execute first rebalance:");
  console.log("   python src/execution/rebalance.py");
  console.log("");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
