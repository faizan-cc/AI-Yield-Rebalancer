const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const network = hre.network.name;
  
  console.log(`\n🔍 Verifying contracts on ${network}...\n`);

  // Load deployment info
  const deploymentPath = path.join(__dirname, `../deployments/${network}_deployment.json`);
  
  if (!fs.existsSync(deploymentPath)) {
    console.error("❌ Deployment file not found. Deploy contracts first.");
    process.exit(1);
  }

  const deployment = JSON.parse(fs.readFileSync(deploymentPath, "utf8"));
  const contracts = deployment.contracts;
  const deployer = deployment.deployer;

  // Verify YieldVault
  console.log("Verifying YieldVault...");
  try {
    await hre.run("verify:verify", {
      address: contracts.YieldVault,
      constructorArguments: [
        deployer,
        contracts.StrategyManager,
        contracts.RebalanceExecutor,
        deployer  // treasury
      ],
    });
    console.log("✅ YieldVault verified");
  } catch (error) {
    console.log("⚠️ YieldVault verification failed:", error.message);
  }

  // Verify StrategyManager
  console.log("\nVerifying StrategyManager...");
  try {
    await hre.run("verify:verify", {
      address: contracts.StrategyManager,
      constructorArguments: [deployer],
    });
    console.log("✅ StrategyManager verified");
  } catch (error) {
    console.log("⚠️ StrategyManager verification failed:", error.message);
  }

  // Verify RebalanceExecutor
  console.log("\nVerifying RebalanceExecutor...");
  try {
    await hre.run("verify:verify", {
      address: contracts.RebalanceExecutor,
      constructorArguments: [
        deployer,
        contracts.YieldVault,
        contracts.StrategyManager,
        deployer
      ],
    });
    console.log("✅ RebalanceExecutor verified");
  } catch (error) {
    console.log("⚠️ RebalanceExecutor verification failed:", error.message);
  }

  // Verify AaveAdapter
  console.log("\nVerifying AaveAdapter...");
  try {
    await hre.run("verify:verify", {
      address: contracts.AaveAdapter,
      constructorArguments: [
        deployer,
        deployment.externalAddresses.aavePool
      ],
    });
    console.log("✅ AaveAdapter verified");
  } catch (error) {
    console.log("⚠️ AaveAdapter verification failed:", error.message);
  }

  // Verify UniswapAdapter
  console.log("\nVerifying UniswapAdapter...");
  try {
    await hre.run("verify:verify", {
      address: contracts.UniswapAdapter,
      constructorArguments: [
        deployer,
        deployment.externalAddresses.swapRouter,
        deployment.externalAddresses.quoter
      ],
    });
    console.log("✅ UniswapAdapter verified");
  } catch (error) {
    console.log("⚠️ UniswapAdapter verification failed:", error.message);
  }

  console.log("\n✅ Verification complete!\n");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
