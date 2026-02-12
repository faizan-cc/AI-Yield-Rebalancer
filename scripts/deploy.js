const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("🚀 Starting deployment...\n");

  const [deployer] = await hre.ethers.getSigners();
  const network = hre.network.name;

  console.log("Network:", network);
  console.log("Deployer:", deployer.address);
  console.log("Balance:", hre.ethers.formatEther(await hre.ethers.provider.getBalance(deployer.address)), "ETH\n");

  // Contract addresses (update for mainnet)
  const ADDRESSES = {
    sepolia: {
      aavePool: "0x6Ae43d3271ff6888e7Fc43Fd7321a503ff738951", // Aave V3 Sepolia
      swapRouter: "0xE592427A0AEce92De3Edee1F18E0157C05861564", // Uniswap V3 Router
      quoter: "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6" // Uniswap V3 Quoter
    },
    base_sepolia: {
      aavePool: "0x07eA79F68B2B3df564D0A34F8e19D9B1e339814b", // Aave V3 Base Sepolia
      swapRouter: "0x94cC0AaC535CCDB3C01d6787D6413C739ae12bc4", // Uniswap V3 Router Base
      quoter: "0xC5290058841028F1614F3A6F0F5816cAd0df5E27" // Uniswap V3 Quoter Base
    }
  };

  const addresses = ADDRESSES[network];
  if (!addresses) {
    throw new Error(`Network ${network} not configured`);
  }

  // Deploy StrategyManager
  console.log("📋 Deploying StrategyManager...");
  const StrategyManager = await hre.ethers.getContractFactory("StrategyManager");
  const strategyManager = await StrategyManager.deploy(deployer.address);
  await strategyManager.waitForDeployment();
  const strategyManagerAddress = await strategyManager.getAddress();
  console.log("✅ StrategyManager deployed to:", strategyManagerAddress);

  // Deploy YieldVault
  console.log("\n🏦 Deploying YieldVault...");
  const YieldVault = await hre.ethers.getContractFactory("YieldVault");
  const yieldVault = await YieldVault.deploy(
    deployer.address,           // initialOwner
    strategyManagerAddress,     // strategyManager  
    deployer.address,          // rebalanceExecutor (deployer for now, RebalanceExecutor will be authorized)
    deployer.address           // treasury
  );
  await yieldVault.waitForDeployment();
  const yieldVaultAddress = await yieldVault.getAddress();
  console.log("✅ YieldVault deployed to:", yieldVaultAddress);

  // Deploy RebalanceExecutor
  console.log("\n⚙️ Deploying RebalanceExecutor...");
  const RebalanceExecutor = await hre.ethers.getContractFactory("RebalanceExecutor");
  const rebalanceExecutor = await RebalanceExecutor.deploy(
    deployer.address,
    yieldVaultAddress,
    strategyManagerAddress,
    deployer.address // keeper = deployer for now
  );
  await rebalanceExecutor.waitForDeployment();
  const rebalanceExecutorAddress = await rebalanceExecutor.getAddress();
  console.log("✅ RebalanceExecutor deployed to:", rebalanceExecutorAddress);

  // Deploy AaveAdapter
  console.log("\n💰 Deploying AaveAdapter...");
  const AaveAdapter = await hre.ethers.getContractFactory("AaveAdapter");
  const aaveAdapter = await AaveAdapter.deploy(
    deployer.address,
    addresses.aavePool
  );
  await aaveAdapter.waitForDeployment();
  const aaveAdapterAddress = await aaveAdapter.getAddress();
  console.log("✅ AaveAdapter deployed to:", aaveAdapterAddress);

  // Deploy UniswapAdapter
  console.log("\n🦄 Deploying UniswapAdapter...");
  const UniswapAdapter = await hre.ethers.getContractFactory("UniswapAdapter");
  const uniswapAdapter = await UniswapAdapter.deploy(
    deployer.address,
    addresses.swapRouter,
    addresses.quoter
  );
  await uniswapAdapter.waitForDeployment();
  const uniswapAdapterAddress = await uniswapAdapter.getAddress();
  console.log("✅ UniswapAdapter deployed to:", uniswapAdapterAddress);

  // Configuration
  console.log("\n⚙️ Configuring contracts...");

  // Set vault in strategy manager
  console.log("Setting vault in StrategyManager...");
  await strategyManager.setVault(yieldVaultAddress);

  // Set ML oracle (deployer for now)
  console.log("Setting ML oracle in StrategyManager...");
  await strategyManager.setMLOracle(deployer.address);

  console.log("✅ Configuration complete!");

  // Save deployment info
  const deployment = {
    network: network,
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      YieldVault: yieldVaultAddress,
      StrategyManager: strategyManagerAddress,
      RebalanceExecutor: rebalanceExecutorAddress,
      AaveAdapter: aaveAdapterAddress,
      UniswapAdapter: uniswapAdapterAddress
    },
    externalAddresses: addresses
  };

  const deploymentPath = path.join(__dirname, `../deployments/${network}_deployment.json`);
  fs.mkdirSync(path.dirname(deploymentPath), { recursive: true });
  fs.writeFileSync(deploymentPath, JSON.stringify(deployment, null, 2));

  console.log("\n📄 Deployment info saved to:", deploymentPath);

  // Summary
  console.log("\n" + "=".repeat(60));
  console.log("🎉 DEPLOYMENT COMPLETE!");
  console.log("=".repeat(60));
  console.log("\n📝 Contract Addresses:");
  console.log("YieldVault:         ", yieldVaultAddress);
  console.log("StrategyManager:    ", strategyManagerAddress);
  console.log("RebalanceExecutor:  ", rebalanceExecutorAddress);
  console.log("AaveAdapter:        ", aaveAdapterAddress);
  console.log("UniswapAdapter:     ", uniswapAdapterAddress);
  
  console.log("\n🔍 Verify contracts with:");
  console.log(`npm run verify:${network}`);
  
  console.log("\n📋 Next steps:");
  console.log("1. Add testnet tokens to vault");
  console.log("2. Register pools in StrategyManager");
  console.log("3. Update ML oracle with pool data");
  console.log("4. Execute first rebalance");
  console.log("\n✨ Happy testing!\n");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
