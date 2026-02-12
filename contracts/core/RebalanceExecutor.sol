// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

interface IYieldVault {
    function rebalance(
        address[] calldata assets,
        uint256[] calldata newAllocations,
        bytes calldata swapData
    ) external;
    
    function getTotalValue() external view returns (uint256);
    function getPortfolio() external view returns (address[] memory, uint256[] memory);
}

interface IStrategyManager {
    function calculateOptimalAllocation() external view returns (
        address[] memory tokens,
        address[] memory protocols,
        uint256[] memory weights,
        uint256 timestamp,
        uint256 expectedAPY,
        string memory reason
    );
    
    function config() external view returns (
        uint8 strategyType,
        uint256 minAPY,
        uint256 maxRiskScore,
        uint256 minTVL,
        uint256 maxPositions,
        uint256 rebalanceThreshold,
        bool requireStablecoin,
        bool isActive
    );
}

/**
 * @title RebalanceExecutor
 * @notice Automated executor for portfolio rebalancing
 * @dev Keeper service calls this to trigger rebalances when conditions are met
 */
contract RebalanceExecutor is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    // Rebalance record
    struct RebalanceRecord {
        uint256 timestamp;
        address[] assetsFrom;
        address[] assetsTo;
        uint256[] amountsFrom;
        uint256[] amountsTo;
        uint256 portfolioValueBefore;
        uint256 portfolioValueAfter;
        uint256 gasUsed;
        string reason;
    }

    // State variables
    address public vault;                      // YieldVault address
    address public strategyManager;            // StrategyManager address
    address public keeper;                     // Authorized keeper address
    uint256 public lastRebalanceTime;         // Last rebalance timestamp
    uint256 public rebalanceCount;            // Total rebalances executed
    uint256 public minRebalanceInterval;      // Minimum time between rebalances (seconds)
    uint256 public maxGasPrice;               // Maximum acceptable gas price (wei)
    bool public autoRebalanceEnabled;         // Whether auto-rebalancing is enabled
    
    mapping(uint256 => RebalanceRecord) public rebalanceHistory; // rebalanceId => record

    // Events
    event RebalanceExecuted(
        uint256 indexed rebalanceId,
        uint256 portfolioValueBefore,
        uint256 portfolioValueAfter,
        uint256 gasUsed,
        string reason
    );
    event KeeperUpdated(address indexed oldKeeper, address indexed newKeeper);
    event VaultUpdated(address indexed oldVault, address indexed newVault);
    event StrategyManagerUpdated(address indexed oldManager, address indexed newManager);
    event AutoRebalanceToggled(bool enabled);
    event MinRebalanceIntervalUpdated(uint256 oldInterval, uint256 newInterval);
    event MaxGasPriceUpdated(uint256 oldMaxGas, uint256 newMaxGas);

    // Modifiers
    modifier onlyKeeper() {
        require(msg.sender == keeper || msg.sender == owner(), "Only keeper or owner");
        _;
    }

    constructor(
        address initialOwner,
        address _vault,
        address _strategyManager,
        address _keeper
    ) Ownable(initialOwner) {
        require(_vault != address(0), "Invalid vault");
        require(_strategyManager != address(0), "Invalid strategy manager");
        require(_keeper != address(0), "Invalid keeper");

        vault = _vault;
        strategyManager = _strategyManager;
        keeper = _keeper;
        minRebalanceInterval = 6 hours;        // Default: rebalance max every 6 hours
        maxGasPrice = 50 gwei;                 // Default: max 50 gwei
        autoRebalanceEnabled = true;
        lastRebalanceTime = block.timestamp;
    }

    /**
     * @notice Check if rebalance should be executed
     * @return _shouldRebalance Whether rebalance should happen
     * @return reason Reason for the decision
     */
    function shouldRebalance() public view returns (bool _shouldRebalance, string memory reason) {
        // Check if auto-rebalance is enabled
        if (!autoRebalanceEnabled) {
            return (false, "Auto-rebalance disabled");
        }

        // Check minimum interval
        if (block.timestamp < lastRebalanceTime + minRebalanceInterval) {
            return (false, "Too soon since last rebalance");
        }

        // Check gas price
        if (tx.gasprice > maxGasPrice) {
            return (false, "Gas price too high");
        }

        // Get current and optimal allocations
        (address[] memory currentTokens, uint256[] memory currentWeights) = IYieldVault(vault).getPortfolio();
        
        (
            address[] memory optimalTokens,
            ,
            uint256[] memory optimalWeights,
            ,
            ,
        ) = IStrategyManager(strategyManager).calculateOptimalAllocation();

        // Check if allocations have changed significantly
        uint256 deviation = _calculateDeviation(
            currentTokens,
            currentWeights,
            optimalTokens,
            optimalWeights
        );

        // Get rebalance threshold from strategy manager
        (
            ,
            ,
            ,
            ,
            ,
            uint256 rebalanceThreshold,
            ,
        ) = IStrategyManager(strategyManager).config();

        if (deviation >= rebalanceThreshold) {
            return (true, "Allocation deviation exceeds threshold");
        }

        return (false, "No significant deviation");
    }

    /**
     * @notice Execute rebalance
     * @dev Called by keeper service
     */
    function executeRebalance() external onlyKeeper nonReentrant returns (uint256 rebalanceId) {
        uint256 startGas = gasleft();

        // Check if rebalance should be executed
        (bool should, string memory reason) = shouldRebalance();
        require(should, reason);

        // Get portfolio value before rebalance
        uint256 portfolioValueBefore = IYieldVault(vault).getTotalValue();

        // Get current portfolio
        (address[] memory currentTokens, uint256[] memory currentWeights) = IYieldVault(vault).getPortfolio();

        // Get optimal allocation from strategy manager
        (
            address[] memory optimalTokens,
            ,
            uint256[] memory optimalWeights,
            ,
            ,
            string memory allocationReason
        ) = IStrategyManager(strategyManager).calculateOptimalAllocation();

        // Execute rebalance on vault
        // Note: In production, swapData would come from 1inch or Paraswap aggregator
        bytes memory swapData = ""; // Empty for now
        IYieldVault(vault).rebalance(optimalTokens, optimalWeights, swapData);

        // Get portfolio value after rebalance
        uint256 portfolioValueAfter = IYieldVault(vault).getTotalValue();

        // Calculate gas used
        uint256 gasUsed = startGas - gasleft();

        // Record rebalance
        rebalanceId = ++rebalanceCount;
        rebalanceHistory[rebalanceId] = RebalanceRecord({
            timestamp: block.timestamp,
            assetsFrom: currentTokens,
            assetsTo: optimalTokens,
            amountsFrom: currentWeights,
            amountsTo: optimalWeights,
            portfolioValueBefore: portfolioValueBefore,
            portfolioValueAfter: portfolioValueAfter,
            gasUsed: gasUsed,
            reason: allocationReason
        });

        lastRebalanceTime = block.timestamp;

        emit RebalanceExecuted(
            rebalanceId,
            portfolioValueBefore,
            portfolioValueAfter,
            gasUsed,
            allocationReason
        );

        return rebalanceId;
    }

    /**
     * @notice Force rebalance regardless of conditions (emergency)
     * @dev Only owner can force rebalance
     */
    function forceRebalance(string calldata reason) external onlyOwner nonReentrant returns (uint256 rebalanceId) {
        uint256 startGas = gasleft();

        // Get portfolio value before rebalance
        uint256 portfolioValueBefore = IYieldVault(vault).getTotalValue();

        // Get current portfolio
        (address[] memory currentTokens, uint256[] memory currentWeights) = IYieldVault(vault).getPortfolio();

        // Get optimal allocation from strategy manager
        (
            address[] memory optimalTokens,
            ,
            uint256[] memory optimalWeights,
            ,
            ,
        ) = IStrategyManager(strategyManager).calculateOptimalAllocation();

        // Execute rebalance on vault
        bytes memory swapData = "";
        IYieldVault(vault).rebalance(optimalTokens, optimalWeights, swapData);

        // Get portfolio value after rebalance
        uint256 portfolioValueAfter = IYieldVault(vault).getTotalValue();

        // Calculate gas used
        uint256 gasUsed = startGas - gasleft();

        // Record rebalance
        rebalanceId = ++rebalanceCount;
        rebalanceHistory[rebalanceId] = RebalanceRecord({
            timestamp: block.timestamp,
            assetsFrom: currentTokens,
            assetsTo: optimalTokens,
            amountsFrom: currentWeights,
            amountsTo: optimalWeights,
            portfolioValueBefore: portfolioValueBefore,
            portfolioValueAfter: portfolioValueAfter,
            gasUsed: gasUsed,
            reason: string(abi.encodePacked("FORCED: ", reason))
        });

        lastRebalanceTime = block.timestamp;

        emit RebalanceExecuted(
            rebalanceId,
            portfolioValueBefore,
            portfolioValueAfter,
            gasUsed,
            reason
        );

        return rebalanceId;
    }

    /**
     * @notice Calculate deviation between current and optimal allocation
     * @dev Returns sum of absolute differences in weights
     */
    function _calculateDeviation(
        address[] memory currentTokens,
        uint256[] memory currentWeights,
        address[] memory optimalTokens,
        uint256[] memory optimalWeights
    ) internal pure returns (uint256 deviation) {
        // Simple approach: sum absolute differences
        // In production, use more sophisticated comparison
        
        uint256 totalDeviation = 0;

        // Check each optimal allocation
        for (uint256 i = 0; i < optimalTokens.length; i++) {
            bool found = false;
            for (uint256 j = 0; j < currentTokens.length; j++) {
                if (optimalTokens[i] == currentTokens[j]) {
                    // Token exists in both, calculate weight difference
                    uint256 diff = optimalWeights[i] > currentWeights[j] 
                        ? optimalWeights[i] - currentWeights[j]
                        : currentWeights[j] - optimalWeights[i];
                    totalDeviation += diff;
                    found = true;
                    break;
                }
            }
            if (!found) {
                // Token not in current portfolio, add full weight
                totalDeviation += optimalWeights[i];
            }
        }

        // Check for tokens in current but not in optimal
        for (uint256 i = 0; i < currentTokens.length; i++) {
            bool found = false;
            for (uint256 j = 0; j < optimalTokens.length; j++) {
                if (currentTokens[i] == optimalTokens[j]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                // Token should be removed, add current weight
                totalDeviation += currentWeights[i];
            }
        }

        return totalDeviation;
    }

    /**
     * @notice Get rebalance record
     */
    function getRebalanceRecord(uint256 rebalanceId) external view returns (RebalanceRecord memory) {
        require(rebalanceId > 0 && rebalanceId <= rebalanceCount, "Invalid rebalance ID");
        return rebalanceHistory[rebalanceId];
    }

    /**
     * @notice Get last N rebalance records
     */
    function getRecentRebalances(uint256 count) external view returns (RebalanceRecord[] memory) {
        uint256 actualCount = count > rebalanceCount ? rebalanceCount : count;
        RebalanceRecord[] memory records = new RebalanceRecord[](actualCount);

        for (uint256 i = 0; i < actualCount; i++) {
            records[i] = rebalanceHistory[rebalanceCount - i];
        }

        return records;
    }

    /**
     * @notice Update keeper address
     */
    function setKeeper(address _keeper) external onlyOwner {
        require(_keeper != address(0), "Invalid keeper");
        address oldKeeper = keeper;
        keeper = _keeper;
        emit KeeperUpdated(oldKeeper, _keeper);
    }

    /**
     * @notice Update vault address
     */
    function setVault(address _vault) external onlyOwner {
        require(_vault != address(0), "Invalid vault");
        address oldVault = vault;
        vault = _vault;
        emit VaultUpdated(oldVault, _vault);
    }

    /**
     * @notice Update strategy manager address
     */
    function setStrategyManager(address _strategyManager) external onlyOwner {
        require(_strategyManager != address(0), "Invalid strategy manager");
        address oldManager = strategyManager;
        strategyManager = _strategyManager;
        emit StrategyManagerUpdated(oldManager, _strategyManager);
    }

    /**
     * @notice Toggle auto-rebalance
     */
    function toggleAutoRebalance(bool enabled) external onlyOwner {
        autoRebalanceEnabled = enabled;
        emit AutoRebalanceToggled(enabled);
    }

    /**
     * @notice Update minimum rebalance interval
     */
    function setMinRebalanceInterval(uint256 interval) external onlyOwner {
        require(interval >= 1 hours, "Interval too short");
        require(interval <= 30 days, "Interval too long");
        uint256 oldInterval = minRebalanceInterval;
        minRebalanceInterval = interval;
        emit MinRebalanceIntervalUpdated(oldInterval, interval);
    }

    /**
     * @notice Update maximum gas price
     */
    function setMaxGasPrice(uint256 gasPrice) external onlyOwner {
        require(gasPrice >= 1 gwei, "Gas price too low");
        require(gasPrice <= 1000 gwei, "Gas price too high");
        uint256 oldMaxGas = maxGasPrice;
        maxGasPrice = gasPrice;
        emit MaxGasPriceUpdated(oldMaxGas, gasPrice);
    }

    /**
     * @notice Get time until next rebalance is allowed
     */
    function timeUntilNextRebalance() external view returns (uint256) {
        uint256 nextTime = lastRebalanceTime + minRebalanceInterval;
        if (block.timestamp >= nextTime) {
            return 0;
        }
        return nextTime - block.timestamp;
    }
}
