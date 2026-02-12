// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title AaveAdapter
 * @notice Adapter for interacting with Aave V3 lending protocol
 * @dev Handles deposits, withdrawals, and yield tracking for Aave positions
 */

// Aave V3 Pool interface
interface IPool {
    function supply(address asset, uint256 amount, address onBehalfOf, uint16 referralCode) external;
    function withdraw(address asset, uint256 amount, address to) external returns (uint256);
    function getReserveData(address asset) external view returns (ReserveData memory);
}

struct ReserveData {
    uint256 configuration;
    uint128 liquidityIndex;
    uint128 currentLiquidityRate;
    uint128 variableBorrowIndex;
    uint128 currentVariableBorrowRate;
    uint128 currentStableBorrowRate;
    uint40 lastUpdateTimestamp;
    uint16 id;
    address aTokenAddress;
    address stableDebtTokenAddress;
    address variableDebtTokenAddress;
    address interestRateStrategyAddress;
    uint128 accruedToTreasury;
    uint128 unbacked;
    uint128 isolationModeTotalDebt;
}

// AToken interface
interface IAToken is IERC20 {
    function UNDERLYING_ASSET_ADDRESS() external view returns (address);
    function balanceOf(address user) external view returns (uint256);
}

contract AaveAdapter is Ownable {
    using SafeERC20 for IERC20;

    // Aave V3 Pool address
    IPool public immutable aavePool;

    // Mapping from underlying token to aToken
    mapping(address => address) public aTokens;

    // Events
    event Deposited(address indexed token, uint256 amount, address indexed aToken);
    event Withdrawn(address indexed token, uint256 amount, address indexed aToken);
    event ATokenRegistered(address indexed token, address indexed aToken);

    constructor(address initialOwner, address _aavePool) Ownable(initialOwner) {
        require(_aavePool != address(0), "Invalid Aave pool");
        aavePool = IPool(_aavePool);
    }

    /**
     * @notice Deposit tokens into Aave V3
     * @param token Underlying token address (e.g., USDC)
     * @param amount Amount to deposit
     * @return aTokenAmount Amount of aTokens received
     */
    function deposit(address token, uint256 amount) external onlyOwner returns (uint256 aTokenAmount) {
        require(token != address(0), "Invalid token");
        require(amount > 0, "Amount must be > 0");

        // Get aToken address
        address aToken = _getAToken(token);
        require(aToken != address(0), "AToken not found");

        // Get aToken balance before
        uint256 aTokenBalanceBefore = IAToken(aToken).balanceOf(address(this));

        // Approve Aave pool
        IERC20(token).safeIncreaseAllowance(address(aavePool), amount);

        // Supply to Aave (referral code 0)
        aavePool.supply(token, amount, address(this), 0);

        // Get aToken balance after
        uint256 aTokenBalanceAfter = IAToken(aToken).balanceOf(address(this));
        aTokenAmount = aTokenBalanceAfter - aTokenBalanceBefore;

        emit Deposited(token, amount, aToken);

        return aTokenAmount;
    }

    /**
     * @notice Withdraw tokens from Aave V3
     * @param token Underlying token address
     * @param amount Amount to withdraw (use type(uint256).max for full balance)
     * @return withdrawnAmount Actual amount withdrawn
     */
    function withdraw(address token, uint256 amount) external onlyOwner returns (uint256 withdrawnAmount) {
        require(token != address(0), "Invalid token");

        // Get aToken address
        address aToken = _getAToken(token);
        require(aToken != address(0), "AToken not found");

        // Get balance before
        uint256 balanceBefore = IERC20(token).balanceOf(address(this));

        // Withdraw from Aave
        withdrawnAmount = aavePool.withdraw(token, amount, address(this));

        // Verify withdrawal
        uint256 balanceAfter = IERC20(token).balanceOf(address(this));
        require(balanceAfter >= balanceBefore + withdrawnAmount, "Withdrawal failed");

        emit Withdrawn(token, withdrawnAmount, aToken);

        return withdrawnAmount;
    }

    /**
     * @notice Get balance of aTokens (including accrued interest)
     * @param token Underlying token address
     * @return balance Balance of aTokens
     */
    function getBalance(address token) external view returns (uint256 balance) {
        address aToken = aTokens[token];
        if (aToken == address(0)) {
            aToken = _getATokenFromReserve(token);
        }
        
        if (aToken == address(0)) {
            return 0;
        }

        return IAToken(aToken).balanceOf(address(this));
    }

    /**
     * @notice Get current APY for a token
     * @param token Underlying token address
     * @return apy Current liquidity rate (APY in ray units: 1e27 = 100%)
     */
    function getCurrentAPY(address token) external view returns (uint256 apy) {
        ReserveData memory data = aavePool.getReserveData(token);
        
        // currentLiquidityRate is in ray (1e27 = 100%)
        // Convert to basis points (1 bps = 0.01%)
        apy = uint256(data.currentLiquidityRate) / 1e23; // Convert ray to bps
        
        return apy;
    }

    /**
     * @notice Get aToken address for underlying token
     * @param token Underlying token address
     * @return aToken AToken address
     */
    function getAToken(address token) external view returns (address aToken) {
        return _getAToken(token);
    }

    /**
     * @notice Register aToken for an underlying token
     * @dev Useful for caching to save gas
     * @param token Underlying token address
     */
    function registerAToken(address token) external onlyOwner {
        require(token != address(0), "Invalid token");
        
        address aToken = _getATokenFromReserve(token);
        require(aToken != address(0), "AToken not found");
        
        aTokens[token] = aToken;
        
        emit ATokenRegistered(token, aToken);
    }

    /**
     * @notice Internal: Get aToken address
     * @dev First checks cache, then queries Aave
     */
    function _getAToken(address token) internal view returns (address) {
        // Check cache first
        address aToken = aTokens[token];
        if (aToken != address(0)) {
            return aToken;
        }

        // Query from Aave reserve data
        return _getATokenFromReserve(token);
    }

    /**
     * @notice Internal: Query aToken from Aave reserve
     */
    function _getATokenFromReserve(address token) internal view returns (address) {
        try aavePool.getReserveData(token) returns (ReserveData memory data) {
            return data.aTokenAddress;
        } catch {
            return address(0);
        }
    }

    /**
     * @notice Get total value including accrued yield
     * @param token Underlying token address
     * @return value Total value in underlying token
     */
    function getTotalValue(address token) external view returns (uint256 value) {
        address aToken = _getAToken(token);
        if (aToken == address(0)) {
            return 0;
        }

        // aToken balance represents underlying + interest
        return IAToken(aToken).balanceOf(address(this));
    }

    /**
     * @notice Emergency withdraw all tokens
     * @dev Only owner can call in emergency
     */
    function emergencyWithdraw(address token, address to) external onlyOwner {
        require(to != address(0), "Invalid recipient");
        
        address aToken = _getAToken(token);
        if (aToken != address(0)) {
            uint256 balance = IAToken(aToken).balanceOf(address(this));
            if (balance > 0) {
                aavePool.withdraw(token, type(uint256).max, to);
            }
        }
    }

    /**
     * @notice Rescue stuck tokens
     * @dev For tokens accidentally sent to this contract
     */
    function rescueTokens(address token, address to, uint256 amount) external onlyOwner {
        require(to != address(0), "Invalid recipient");
        IERC20(token).safeTransfer(to, amount);
    }
}
