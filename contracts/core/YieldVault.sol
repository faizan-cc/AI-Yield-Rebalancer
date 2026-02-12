// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

/**
 * @title YieldVault
 * @notice Main vault contract for AI-driven DeFi yield optimization
 * @dev ERC-4626 compatible vault with multi-strategy support
 * 
 * Features:
 * - Multi-asset portfolio management
 * - Strategy-based rebalancing
 * - Emergency pause mechanism
 * - Performance fee collection
 * - Gas-optimized operations
 */
contract YieldVault is ReentrancyGuard, Ownable, Pausable {
    using SafeERC20 for IERC20;

    // ============ State Variables ============

    /// @notice Vault name
    string public constant NAME = "AI DeFi Yield Vault";
    
    /// @notice Vault version
    string public constant VERSION = "1.0.0";

    /// @notice Management fee (0.5% annually = 50 basis points)
    uint256 public constant MANAGEMENT_FEE_BPS = 50;
    
    /// @notice Performance fee (10% of profits)
    uint256 public constant PERFORMANCE_FEE_BPS = 1000;
    
    /// @notice Basis points denominator
    uint256 public constant BPS_DENOMINATOR = 10000;

    /// @notice Strategy manager contract
    address public strategyManager;
    
    /// @notice Rebalance executor contract
    address public rebalanceExecutor;
    
    /// @notice Treasury address for fee collection
    address public treasury;

    /// @notice Total value locked in vault
    uint256 public totalValueLocked;
    
    /// @notice Last rebalance timestamp
    uint256 public lastRebalanceTime;
    
    /// @notice Rebalance frequency (5 minutes for testnet testing)
    uint256 public constant REBALANCE_FREQUENCY = 5 minutes;

    /// @notice Asset allocations: asset address => amount
    mapping(address => uint256) public allocations;
    
    /// @notice Supported assets list
    address[] public supportedAssets;
    
    /// @notice User shares: user address => share amount
    mapping(address => uint256) public shares;
    
    /// @notice Total shares issued
    uint256 public totalShares;

    /// @notice Historical high water mark for performance fees
    uint256 public highWaterMark;

    // ============ Events ============

    event Deposit(address indexed user, address indexed asset, uint256 amount, uint256 shares);
    event Withdraw(address indexed user, address indexed asset, uint256 amount, uint256 shares);
    event Rebalance(address indexed executor, uint256 timestamp, uint256 newTVL);
    event FeeCollected(uint256 managementFee, uint256 performanceFee);
    event StrategyManagerUpdated(address indexed oldManager, address indexed newManager);
    event AssetAllocated(address indexed asset, uint256 amount);

    // ============ Modifiers ============

    modifier onlyStrategyManager() {
        require(msg.sender == strategyManager, "Only strategy manager");
        _;
    }

    modifier onlyRebalanceExecutor() {
        require(msg.sender == rebalanceExecutor, "Only rebalance executor");
        _;
    }

    // ============ Constructor ============

    constructor(
        address initialOwner,
        address _strategyManager,
        address _rebalanceExecutor,
        address _treasury
    ) Ownable(initialOwner) {
        require(_strategyManager != address(0), "Invalid strategy manager");
        require(_rebalanceExecutor != address(0), "Invalid rebalance executor");
        require(_treasury != address(0), "Invalid treasury");

        strategyManager = _strategyManager;
        rebalanceExecutor = _rebalanceExecutor;
        treasury = _treasury;
        lastRebalanceTime = block.timestamp;
        highWaterMark = 0;
    }

    // ============ User Functions ============

    /**
     * @notice Deposit assets into vault
     * @param asset Asset address to deposit
     * @param amount Amount to deposit
     * @return sharesIssued Number of shares issued
     */
    function deposit(address asset, uint256 amount) 
        external 
        nonReentrant 
        whenNotPaused 
        returns (uint256 sharesIssued) 
    {
        require(amount > 0, "Amount must be > 0");
        require(isAssetSupported(asset), "Asset not supported");

        // Transfer tokens from user
        IERC20(asset).safeTransferFrom(msg.sender, address(this), amount);

        // Calculate shares to issue
        if (totalShares == 0) {
            // First deposit: 1:1 ratio
            sharesIssued = amount;
        } else {
            // Calculate shares based on current TVL
            sharesIssued = (amount * totalShares) / totalValueLocked;
        }

        // Update state
        shares[msg.sender] += sharesIssued;
        totalShares += sharesIssued;
        allocations[asset] += amount;
        totalValueLocked += amount;

        emit Deposit(msg.sender, asset, amount, sharesIssued);
        return sharesIssued;
    }

    /**
     * @notice Withdraw assets from vault
     * @param sharesToBurn Number of shares to burn
     * @return assets Array of asset addresses withdrawn
     * @return amounts Array of amounts withdrawn
     */
    function withdraw(uint256 sharesToBurn) 
        external 
        nonReentrant 
        returns (address[] memory assets, uint256[] memory amounts) 
    {
        require(sharesToBurn > 0, "Shares must be > 0");
        require(shares[msg.sender] >= sharesToBurn, "Insufficient shares");

        // Calculate withdrawal proportion
        uint256 proportion = (sharesToBurn * 1e18) / totalShares;

        // Prepare arrays
        assets = supportedAssets;
        amounts = new uint256[](supportedAssets.length);

        // Calculate and transfer proportional amounts
        uint256 totalWithdrawn = 0;
        for (uint256 i = 0; i < supportedAssets.length; i++) {
            address asset = supportedAssets[i];
            uint256 amount = (allocations[asset] * proportion) / 1e18;
            
            if (amount > 0) {
                amounts[i] = amount;
                allocations[asset] -= amount;
                totalWithdrawn += amount;
                IERC20(asset).safeTransfer(msg.sender, amount);
                
                emit Withdraw(msg.sender, asset, amount, sharesToBurn);
            }
        }

        // Update state
        shares[msg.sender] -= sharesToBurn;
        totalShares -= sharesToBurn;
        totalValueLocked -= totalWithdrawn;

        return (assets, amounts);
    }

    // ============ Strategy Functions ============

    /**
     * @notice Execute rebalancing based on strategy
     * @param newAllocations Array of new allocation targets
     */
    function rebalance(
        address[] calldata assets,
        uint256[] calldata newAllocations
    ) external onlyRebalanceExecutor whenNotPaused {
        require(assets.length == newAllocations.length, "Length mismatch");
        require(
            block.timestamp >= lastRebalanceTime + REBALANCE_FREQUENCY,
            "Rebalance too soon"
        );

        // Collect management fees before rebalancing
        _collectManagementFees();

        // Update allocations
        uint256 newTVL = 0;
        for (uint256 i = 0; i < assets.length; i++) {
            allocations[assets[i]] = newAllocations[i];
            newTVL += newAllocations[i];
            emit AssetAllocated(assets[i], newAllocations[i]);
        }

        // Collect performance fees if TVL increased
        if (newTVL > highWaterMark) {
            _collectPerformanceFees(newTVL);
            highWaterMark = newTVL;
        }

        totalValueLocked = newTVL;
        lastRebalanceTime = block.timestamp;

        emit Rebalance(msg.sender, block.timestamp, newTVL);
    }

    /**
     * @notice Add support for new asset
     * @param asset Asset address to add
     */
    function addSupportedAsset(address asset) external onlyOwner {
        require(asset != address(0), "Invalid asset");
        require(!isAssetSupported(asset), "Already supported");
        
        supportedAssets.push(asset);
    }

    // ============ Fee Collection ============

    /**
     * @dev Collect management fees (0.5% annually)
     */
    function _collectManagementFees() private {
        if (totalValueLocked == 0) return;

        uint256 timeElapsed = block.timestamp - lastRebalanceTime;
        uint256 annualFee = (totalValueLocked * MANAGEMENT_FEE_BPS) / BPS_DENOMINATOR;
        uint256 fee = (annualFee * timeElapsed) / 365 days;

        if (fee > 0) {
            // Deduct fee from TVL
            totalValueLocked -= fee;
            emit FeeCollected(fee, 0);
        }
    }

    /**
     * @dev Collect performance fees (10% of profits)
     */
    function _collectPerformanceFees(uint256 currentTVL) private {
        uint256 profit = currentTVL - highWaterMark;
        uint256 fee = (profit * PERFORMANCE_FEE_BPS) / BPS_DENOMINATOR;

        if (fee > 0) {
            totalValueLocked -= fee;
            emit FeeCollected(0, fee);
        }
    }

    // ============ Admin Functions ============

    /**
     * @notice Update strategy manager
     * @param newManager New strategy manager address
     */
    function updateStrategyManager(address newManager) external onlyOwner {
        require(newManager != address(0), "Invalid manager");
        address oldManager = strategyManager;
        strategyManager = newManager;
        emit StrategyManagerUpdated(oldManager, newManager);
    }

    /**
     * @notice Update rebalance executor
     */
    function updateRebalanceExecutor(address newExecutor) external onlyOwner {
        require(newExecutor != address(0), "Invalid executor");
        rebalanceExecutor = newExecutor;
    }

    /**
     * @notice Emergency pause
     */
    function pause() external onlyOwner {
        _pause();
    }

    /**
     * @notice Unpause
     */
    function unpause() external onlyOwner {
        _unpause();
    }

    // ============ View Functions ============

    /**
     * @notice Check if asset is supported
     */
    function isAssetSupported(address asset) public view returns (bool) {
        for (uint256 i = 0; i < supportedAssets.length; i++) {
            if (supportedAssets[i] == asset) return true;
        }
        return false;
    }

    /**
     * @notice Get user's share of TVL
     */
    function getUserValue(address user) external view returns (uint256) {
        if (totalShares == 0) return 0;
        return (shares[user] * totalValueLocked) / totalShares;
    }

    /**
     * @notice Get all allocations
     */
    function getAllocations() external view returns (address[] memory, uint256[] memory) {
        uint256[] memory amounts = new uint256[](supportedAssets.length);
        for (uint256 i = 0; i < supportedAssets.length; i++) {
            amounts[i] = allocations[supportedAssets[i]];
        }
        return (supportedAssets, amounts);
    }

    /**
     * @notice Check if rebalance is due
     */
    function isRebalanceDue() external view returns (bool) {
        return block.timestamp >= lastRebalanceTime + REBALANCE_FREQUENCY;
    }
}
