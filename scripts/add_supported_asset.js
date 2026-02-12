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
  
  const YIELD_VAULT = deployment.contracts.YieldVault;
  const USDC_ADDRESS = "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238";
  
  console.log("\n📋 Adding Faucet USDC as Supported Asset...\n");
  console.log("USDC Address:", USDC_ADDRESS);
  console.log("Vault Address:", YIELD_VAULT);
  
  const vault = await ethers.getContractAt("YieldVault", YIELD_VAULT);
  
  // Add as supported asset (skip check, just add it)
  console.log("\n🔄 Adding USDC as supported asset...");
  
  try {
    const tx = await vault.addSupportedAsset(USDC_ADDRESS);
    console.log("Transaction hash:", tx.hash);
    console.log("Waiting for confirmation...");
    
    await tx.wait();
    console.log("✅ Asset added successfully!");
  } catch (error) {
    if (error.message.includes("Already supported")) {
      console.log("✅ USDC is already supported!");
    } else {
      console.error("❌ Error:", error.message);
      throw error;
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
