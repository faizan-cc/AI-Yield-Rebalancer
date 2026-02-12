// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/**
 * @title StrategyManager
 * @notice Manages strategy selection and ML-driven allocation decisions
 * @dev Receives predictions from off-chain ML models and determines optimal allocations
 */
contract StrategyManager is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    // Strategy types
    enum StrategyType {
        HIGHEST_APY,      // Strategy 1: Always pick highest APY
        TVL_WEIGHTED,     // Strategy 2: Weight by TVL for safety
        OPTIMIZED_ML,     // Strategy 3: ML-optimized allocation
        STABLECOIN_ONLY   // Strategy 4: Stablecoin yield only
    }

    // Pool information
    struct Pool {
        address token;           // Underlying token
        address protocol;        // Protocol adapter address
        string protocolName;     // Protocol name (e.g., "aave_v3")
        uint256 currentAPY;      // Current APY (in basis points, e.g., 500 = 5%)
        uint256 tvl;             // Total Value Locked
        uint256 riskScore;       // Risk score from ML (0-100)
        bool isActive;           // Whether pool is active
        uint256 lastUpdate;      // Last update timestamp
    }

    // Strategy configuration
    struct StrategyConfig {
        StrategyType strategyType;
        uint256 minAPY;              // Minimum acceptable APY (bps)
        uint256 maxRiskScore;        // Maximum acceptable risk score
        uint256 minTVL;              // Minimum TVL requirement
        uint256 maxPositions;        // Maximum number of positions
        uint256 rebalanceThreshold;  // Minimum gain to trigger rebalance (bps)
        bool requireStablecoin;      // Only use stablecoins
        bool isActive;
    }

    // Allocation recommendation
    struct Allocation {
        address[] tokens;
        address[] protocols;
        uint256[] weights;     // Weights in basis points (sum = 10000)
        uint256 timestamp;
        uint256 expectedAPY;   // Expected weighted APY
        string reason;         // Reason for allocation
    }

    // State variables
    mapping(bytes32 => Pool) public pools;           // poolId => Pool
    bytes32[] public poolIds;                         // List of all pool IDs
    StrategyConfig public config;                     // Current strategy configuration
    Allocation public currentAllocation;              // Current allocation
    address public mlOracle;                          // Address authorized to update ML predictions
    address public vault;                             // Vault contract address

    // Events
    event PoolAdded(bytes32 indexed poolId, address token, address protocol, string protocolName);
    event PoolUpdated(bytes32 indexed poolId, uint256 apy, uint256 tvl, uint256 riskScore);
    event PoolDeactivated(bytes32 indexed poolId);
    event StrategyConfigUpdated(StrategyType strategyType);
    event AllocationProposed(address[] tokens, address[] protocols, uint256[] weights, uint256 expectedAPY);
    event MLOracleUpdated(address indexed oldOracle, address indexed newOracle);
    event VaultUpdated(address indexed oldVault, address indexed newVault);

    // Modifiers
    modifier onlyMLOracle() {
        require(msg.sender == mlOracle, "Only ML oracle");
        _;
    }

    modifier onlyVault() {
        require(msg.sender == vault, "Only vault");
        _;
    }

    constructor(address initialOwner) Ownable(initialOwner) {
        // Default configuration: Optimized ML strategy
        config = StrategyConfig({
            strategyType: StrategyType.OPTIMIZED_ML,
            minAPY: 100,              // 1% minimum APY
            maxRiskScore: 70,         // Max risk score of 70
            minTVL: 100000e6,         // $100K minimum TVL
            maxPositions: 5,          // Max 5 positions
            rebalanceThreshold: 100,  // 1% gain needed to rebalance
            requireStablecoin: false,
            isActive: true
        });
    }

    /**
     * @notice Add a new pool to the strategy manager
     * @param token Underlying token address
     * @param protocol Protocol adapter address
     * @param protocolName Protocol name
     */
    function addPool(
        address token,
        address protocol,
        string memory protocolName
    ) external onlyOwner {
        bytes32 poolId = keccak256(abi.encodePacked(token, protocol));
        
        require(pools[poolId].token == address(0), "Pool already exists");
        require(token != address(0), "Invalid token");
        require(protocol != address(0), "Invalid protocol");

        pools[poolId] = Pool({
            token: token,
            protocol: protocol,
            protocolName: protocolName,
            currentAPY: 0,
            tvl: 0,
            riskScore: 50, // Default medium risk
            isActive: true,
            lastUpdate: block.timestamp
        });

        poolIds.push(poolId);

        emit PoolAdded(poolId, token, protocol, protocolName);
    }

    /**
     * @notice Update pool data (APY, TVL, risk score)
     * @dev Called by ML oracle with fresh data
     */
    function updatePool(
        bytes32 poolId,
        uint256 apy,
        uint256 tvl,
        uint256 riskScore
    ) external onlyMLOracle {
        require(pools[poolId].isActive, "Pool not active");
        require(riskScore <= 100, "Invalid risk score");

        Pool storage pool = pools[poolId];
        pool.currentAPY = apy;
        pool.tvl = tvl;
        pool.riskScore = riskScore;
        pool.lastUpdate = block.timestamp;

        emit PoolUpdated(poolId, apy, tvl, riskScore);
    }

    /**
     * @notice Batch update multiple pools
     * @dev More gas efficient for updating all pools at once
     */
    function batchUpdatePools(
        bytes32[] calldata _poolIds,
        uint256[] calldata apys,
        uint256[] calldata tvls,
        uint256[] calldata riskScores
    ) external onlyMLOracle {
        require(_poolIds.length == apys.length, "Length mismatch");
        require(_poolIds.length == tvls.length, "Length mismatch");
        require(_poolIds.length == riskScores.length, "Length mismatch");

        for (uint256 i = 0; i < _poolIds.length; i++) {
            bytes32 poolId = _poolIds[i];
            if (pools[poolId].isActive) {
                pools[poolId].currentAPY = apys[i];
                pools[poolId].tvl = tvls[i];
                pools[poolId].riskScore = riskScores[i];
                pools[poolId].lastUpdate = block.timestamp;

                emit PoolUpdated(poolId, apys[i], tvls[i], riskScores[i]);
            }
        }
    }

    /**
     * @notice Calculate optimal allocation based on current strategy
     * @return Allocation struct with recommended allocations
     */
    function calculateOptimalAllocation() external view returns (Allocation memory) {
        if (config.strategyType == StrategyType.HIGHEST_APY) {
            return _calculateHighestAPY();
        } else if (config.strategyType == StrategyType.TVL_WEIGHTED) {
            return _calculateTVLWeighted();
        } else if (config.strategyType == StrategyType.OPTIMIZED_ML) {
            return _calculateOptimizedML();
        } else if (config.strategyType == StrategyType.STABLECOIN_ONLY) {
            return _calculateStablecoinOnly();
        }
        
        revert("Invalid strategy type");
    }

    /**
     * @notice Strategy 1: Highest APY - Pick top pools by APY
     */
    function _calculateHighestAPY() internal view returns (Allocation memory) {
        // Get all eligible pools
        bytes32[] memory eligible = _getEligiblePools();
        require(eligible.length > 0, "No eligible pools");

        // Sort by APY (descending)
        eligible = _sortByAPY(eligible);

        // Take top N pools
        uint256 numPositions = eligible.length < config.maxPositions ? eligible.length : config.maxPositions;
        
        address[] memory tokens = new address[](numPositions);
        address[] memory protocols = new address[](numPositions);
        uint256[] memory weights = new uint256[](numPositions);
        uint256 totalAPY = 0;

        // Equal weight distribution
        uint256 weightPerPool = 10000 / numPositions;
        uint256 remainder = 10000 % numPositions;

        for (uint256 i = 0; i < numPositions; i++) {
            Pool memory pool = pools[eligible[i]];
            tokens[i] = pool.token;
            protocols[i] = pool.protocol;
            weights[i] = weightPerPool + (i == 0 ? remainder : 0); // Add remainder to first pool
            totalAPY += pool.currentAPY * weights[i] / 10000;
        }

        return Allocation({
            tokens: tokens,
            protocols: protocols,
            weights: weights,
            timestamp: block.timestamp,
            expectedAPY: totalAPY,
            reason: "Highest APY strategy"
        });
    }

    /**
     * @notice Strategy 2: TVL Weighted - Weight by TVL for safety
     */
    function _calculateTVLWeighted() internal view returns (Allocation memory) {
        bytes32[] memory eligible = _getEligiblePools();
        require(eligible.length > 0, "No eligible pools");

        // Calculate total TVL
        uint256 totalTVL = 0;
        for (uint256 i = 0; i < eligible.length; i++) {
            totalTVL += pools[eligible[i]].tvl;
        }

        require(totalTVL > 0, "Zero total TVL");

        // Allocate based on TVL proportions
        uint256 numPositions = eligible.length < config.maxPositions ? eligible.length : config.maxPositions;
        
        address[] memory tokens = new address[](numPositions);
        address[] memory protocols = new address[](numPositions);
        uint256[] memory weights = new uint256[](numPositions);
        uint256 totalWeight = 0;
        uint256 totalAPY = 0;

        for (uint256 i = 0; i < numPositions; i++) {
            Pool memory pool = pools[eligible[i]];
            tokens[i] = pool.token;
            protocols[i] = pool.protocol;
            weights[i] = (pool.tvl * 10000) / totalTVL;
            totalWeight += weights[i];
            totalAPY += pool.currentAPY * weights[i] / 10000;
        }

        // Normalize weights to exactly 10000
        if (totalWeight != 10000 && totalWeight > 0) {
            weights[0] += 10000 - totalWeight;
        }

        return Allocation({
            tokens: tokens,
            protocols: protocols,
            weights: weights,
            timestamp: block.timestamp,
            expectedAPY: totalAPY,
            reason: "TVL weighted for safety"
        });
    }

    /**
     * @notice Strategy 3: Optimized ML - Use ML predictions
     * @dev This uses the current allocation set by ML oracle
     */
    function _calculateOptimizedML() internal view returns (Allocation memory) {
        require(currentAllocation.tokens.length > 0, "No ML allocation set");
        return currentAllocation;
    }

    /**
     * @notice Strategy 4: Stablecoin Only - Only allocate to stablecoin pools
     */
    function _calculateStablecoinOnly() internal view returns (Allocation memory) {
        bytes32[] memory eligible = _getEligiblePools();
        
        // Filter for stablecoins (simplified - in production, use oracle)
        uint256 count = 0;
        for (uint256 i = 0; i < eligible.length; i++) {
            if (_isStablecoin(pools[eligible[i]].token)) {
                count++;
            }
        }

        require(count > 0, "No stablecoin pools");

        // Sort stablecoins by APY
        eligible = _sortByAPY(eligible);

        uint256 numPositions = count < config.maxPositions ? count : config.maxPositions;
        
        address[] memory tokens = new address[](numPositions);
        address[] memory protocols = new address[](numPositions);
        uint256[] memory weights = new uint256[](numPositions);
        uint256 totalAPY = 0;

        uint256 weightPerPool = 10000 / numPositions;
        uint256 remainder = 10000 % numPositions;

        uint256 idx = 0;
        for (uint256 i = 0; i < eligible.length && idx < numPositions; i++) {
            if (_isStablecoin(pools[eligible[i]].token)) {
                Pool memory pool = pools[eligible[i]];
                tokens[idx] = pool.token;
                protocols[idx] = pool.protocol;
                weights[idx] = weightPerPool + (idx == 0 ? remainder : 0);
                totalAPY += pool.currentAPY * weights[idx] / 10000;
                idx++;
            }
        }

        return Allocation({
            tokens: tokens,
            protocols: protocols,
            weights: weights,
            timestamp: block.timestamp,
            expectedAPY: totalAPY,
            reason: "Stablecoin only for stability"
        });
    }

    /**
     * @notice Get all eligible pools based on strategy constraints
     */
    function _getEligiblePools() internal view returns (bytes32[] memory) {
        uint256 count = 0;
        
        // First pass: count eligible
        for (uint256 i = 0; i < poolIds.length; i++) {
            Pool memory pool = pools[poolIds[i]];
            if (_isEligible(pool)) {
                count++;
            }
        }

        bytes32[] memory eligible = new bytes32[](count);
        uint256 idx = 0;

        // Second pass: collect eligible
        for (uint256 i = 0; i < poolIds.length; i++) {
            Pool memory pool = pools[poolIds[i]];
            if (_isEligible(pool)) {
                eligible[idx] = poolIds[i];
                idx++;
            }
        }

        return eligible;
    }

    /**
     * @notice Check if pool is eligible based on strategy config
     */
    function _isEligible(Pool memory pool) internal view returns (bool) {
        return pool.isActive &&
               pool.currentAPY >= config.minAPY &&
               pool.riskScore <= config.maxRiskScore &&
               pool.tvl >= config.minTVL &&
               (!config.requireStablecoin || _isStablecoin(pool.token));
    }

    /**
     * @notice Sort pools by APY (descending)
     */
    function _sortByAPY(bytes32[] memory _poolIds) internal view returns (bytes32[] memory) {
        // Simple bubble sort (ok for small arrays, use quicksort for production)
        for (uint256 i = 0; i < _poolIds.length; i++) {
            for (uint256 j = i + 1; j < _poolIds.length; j++) {
                if (pools[_poolIds[i]].currentAPY < pools[_poolIds[j]].currentAPY) {
                    bytes32 temp = _poolIds[i];
                    _poolIds[i] = _poolIds[j];
                    _poolIds[j] = temp;
                }
            }
        }
        return _poolIds;
    }

    /**
     * @notice Check if token is a stablecoin (simplified)
     */
    function _isStablecoin(address token) internal pure returns (bool) {
        // In production, use Chainlink oracle or whitelist
        // For now, we'll have the ML oracle set this
        return true; // Placeholder
    }

    /**
     * @notice Set ML-optimized allocation (Strategy 3)
     * @dev Called by ML oracle with optimized allocations
     */
    function setMLAllocation(
        address[] calldata tokens,
        address[] calldata protocols,
        uint256[] calldata weights,
        uint256 expectedAPY,
        string calldata reason
    ) external onlyMLOracle {
        require(tokens.length == protocols.length, "Length mismatch");
        require(tokens.length == weights.length, "Length mismatch");
        require(tokens.length > 0, "Empty allocation");
        require(tokens.length <= config.maxPositions, "Too many positions");

        // Validate weights sum to 10000 (100%)
        uint256 totalWeight = 0;
        for (uint256 i = 0; i < weights.length; i++) {
            totalWeight += weights[i];
        }
        require(totalWeight == 10000, "Weights must sum to 10000");

        currentAllocation = Allocation({
            tokens: tokens,
            protocols: protocols,
            weights: weights,
            timestamp: block.timestamp,
            expectedAPY: expectedAPY,
            reason: reason
        });

        emit AllocationProposed(tokens, protocols, weights, expectedAPY);
    }

    /**
     * @notice Update strategy configuration
     */
    function updateStrategyConfig(
        StrategyType strategyType,
        uint256 minAPY,
        uint256 maxRiskScore,
        uint256 minTVL,
        uint256 maxPositions,
        uint256 rebalanceThreshold,
        bool requireStablecoin
    ) external onlyOwner {
        require(maxPositions > 0, "Max positions must be > 0");
        require(maxRiskScore <= 100, "Invalid risk score");

        config = StrategyConfig({
            strategyType: strategyType,
            minAPY: minAPY,
            maxRiskScore: maxRiskScore,
            minTVL: minTVL,
            maxPositions: maxPositions,
            rebalanceThreshold: rebalanceThreshold,
            requireStablecoin: requireStablecoin,
            isActive: true
        });

        emit StrategyConfigUpdated(strategyType);
    }

    /**
     * @notice Set ML oracle address
     */
    function setMLOracle(address _mlOracle) external onlyOwner {
        require(_mlOracle != address(0), "Invalid oracle");
        address oldOracle = mlOracle;
        mlOracle = _mlOracle;
        emit MLOracleUpdated(oldOracle, _mlOracle);
    }

    /**
     * @notice Set vault address
     */
    function setVault(address _vault) external onlyOwner {
        require(_vault != address(0), "Invalid vault");
        address oldVault = vault;
        vault = _vault;
        emit VaultUpdated(oldVault, _vault);
    }

    /**
     * @notice Deactivate a pool
     */
    function deactivatePool(bytes32 poolId) external onlyOwner {
        require(pools[poolId].isActive, "Already inactive");
        pools[poolId].isActive = false;
        emit PoolDeactivated(poolId);
    }

    /**
     * @notice Get pool by ID
     */
    function getPool(bytes32 poolId) external view returns (Pool memory) {
        return pools[poolId];
    }

    /**
     * @notice Get all active pools
     */
    function getActivePools() external view returns (bytes32[] memory) {
        uint256 count = 0;
        for (uint256 i = 0; i < poolIds.length; i++) {
            if (pools[poolIds[i]].isActive) {
                count++;
            }
        }

        bytes32[] memory active = new bytes32[](count);
        uint256 idx = 0;
        for (uint256 i = 0; i < poolIds.length; i++) {
            if (pools[poolIds[i]].isActive) {
                active[idx] = poolIds[i];
                idx++;
            }
        }

        return active;
    }

    /**
     * @notice Get current strategy configuration
     */
    function getStrategyConfig() external view returns (StrategyConfig memory) {
        return config;
    }
}
