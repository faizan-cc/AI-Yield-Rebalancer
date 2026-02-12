const hre = require("hardhat");
const { ethers } = require("hardhat");

async function main() {
  const [signer] = await ethers.getSigners();
  
  const WETH_ADDRESS = "0xC558DBdd856501FCd9aaF1E62eae57A9F0629a3c";
  const amountInEth = "0.5"; // Wrap 0.5 ETH → 0.5 WETH
  
  console.log("🔄 Wrapping ETH to WETH on Sepolia...\n");
  console.log("Account:", signer.address);
  
  const balance = await ethers.provider.getBalance(signer.address);
  console.log("Current ETH:", ethers.formatEther(balance), "ETH");
  console.log("Amount to wrap:", amountInEth, "ETH\n");

  const weth = await ethers.getContractAt(
    [
      "function deposit() payable",
      "function balanceOf(address) view returns (uint256)",
      "function withdraw(uint256) external"
    ],
    WETH_ADDRESS
  );

  // Check current WETH balance
  const wethBalanceBefore = await weth.balanceOf(signer.address);
  console.log("WETH before:", ethers.formatEther(wethBalanceBefore), "WETH");

  // Wrap ETH
  console.log("\n📤 Sending transaction...");
  const tx = await weth.deposit({ 
    value: ethers.parseEther(amountInEth),
    gasLimit: 100000 
  });
  
  console.log("Transaction hash:", tx.hash);
  console.log("Waiting for confirmation...");
  
  await tx.wait();
  console.log("✅ Transaction confirmed!");

  // Check new balance
  const wethBalanceAfter = await weth.balanceOf(signer.address);
  console.log("\n💰 WETH after:", ethers.formatEther(wethBalanceAfter), "WETH");
  console.log("Wrapped:", ethers.formatEther(wethBalanceAfter - wethBalanceBefore), "WETH");
  
  const ethBalanceAfter = await ethers.provider.getBalance(signer.address);
  console.log("\nRemaining ETH:", ethers.formatEther(ethBalanceAfter), "ETH");
  
  console.log("\n✅ Success! You now have WETH for testing.");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
