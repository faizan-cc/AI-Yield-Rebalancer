// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title UniswapAdapter
 * @notice Adapter for swapping tokens via Uniswap V3
 * @dev Used during rebalancing to swap between different yield positions
 */

// Uniswap V3 Router interface
interface ISwapRouter {
    struct ExactInputSingleParams {
        address tokenIn;
        address tokenOut;
        uint24 fee;
        address recipient;
        uint256 deadline;
        uint256 amountIn;
        uint256 amountOutMinimum;
        uint160 sqrtPriceLimitX96;
    }

    function exactInputSingle(ExactInputSingleParams calldata params) external payable returns (uint256 amountOut);

    struct ExactInputParams {
        bytes path;
        address recipient;
        uint256 deadline;
        uint256 amountIn;
        uint256 amountOutMinimum;
    }

    function exactInput(ExactInputParams calldata params) external payable returns (uint256 amountOut);
}

// Uniswap V3 Quoter interface
interface IQuoter {
    function quoteExactInputSingle(
        address tokenIn,
        address tokenOut,
        uint24 fee,
        uint256 amountIn,
        uint160 sqrtPriceLimitX96
    ) external returns (uint256 amountOut);
}

contract UniswapAdapter is Ownable {
    using SafeERC20 for IERC20;

    // Uniswap V3 Router
    ISwapRouter public immutable swapRouter;
    
    // Uniswap V3 Quoter
    IQuoter public immutable quoter;

    // Default fee tier (0.3%)
    uint24 public constant DEFAULT_FEE = 3000;

    // Maximum slippage in basis points (default 1% = 100 bps)
    uint256 public maxSlippageBps;

    // Events
    event Swapped(
        address indexed tokenIn,
        address indexed tokenOut,
        uint256 amountIn,
        uint256 amountOut,
        uint256 slippage
    );
    event MaxSlippageUpdated(uint256 oldSlippage, uint256 newSlippage);

    constructor(
        address initialOwner,
        address _swapRouter,
        address _quoter
    ) Ownable(initialOwner) {
        require(_swapRouter != address(0), "Invalid swap router");
        require(_quoter != address(0), "Invalid quoter");
        
        swapRouter = ISwapRouter(_swapRouter);
        quoter = IQuoter(_quoter);
        maxSlippageBps = 100; // 1% default
    }

    /**
     * @notice Swap tokens using Uniswap V3
     * @param tokenIn Input token address
     * @param tokenOut Output token address
     * @param amountIn Amount of input token
     * @param minAmountOut Minimum amount of output token (slippage protection)
     * @param fee Pool fee tier (500 = 0.05%, 3000 = 0.3%, 10000 = 1%)
     * @return amountOut Actual amount of output tokens received
     */
    function swap(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 minAmountOut,
        uint24 fee
    ) external onlyOwner returns (uint256 amountOut) {
        require(tokenIn != address(0), "Invalid tokenIn");
        require(tokenOut != address(0), "Invalid tokenOut");
        require(amountIn > 0, "Amount must be > 0");

        // Approve router
        IERC20(tokenIn).safeIncreaseAllowance(address(swapRouter), amountIn);

        // Prepare swap params
        ISwapRouter.ExactInputSingleParams memory params = ISwapRouter.ExactInputSingleParams({
            tokenIn: tokenIn,
            tokenOut: tokenOut,
            fee: fee,
            recipient: address(this),
            deadline: block.timestamp,
            amountIn: amountIn,
            amountOutMinimum: minAmountOut,
            sqrtPriceLimitX96: 0 // No price limit
        });

        // Execute swap
        amountOut = swapRouter.exactInputSingle(params);

        // Calculate slippage
        uint256 slippage = 0;
        if (minAmountOut > 0) {
            slippage = ((amountOut - minAmountOut) * 10000) / minAmountOut;
        }

        emit Swapped(tokenIn, tokenOut, amountIn, amountOut, slippage);

        return amountOut;
    }

    /**
     * @notice Swap tokens with automatic slippage calculation
     * @param tokenIn Input token address
     * @param tokenOut Output token address
     * @param amountIn Amount of input token
     * @param fee Pool fee tier
     * @return amountOut Actual amount of output tokens received
     */
    function swapWithAutoSlippage(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint24 fee
    ) external onlyOwner returns (uint256 amountOut) {
        // Get quote
        uint256 expectedOut = getQuote(tokenIn, tokenOut, amountIn, fee);
        
        // Calculate minimum amount with max slippage
        uint256 minAmountOut = (expectedOut * (10000 - maxSlippageBps)) / 10000;

        // Approve router
        IERC20(tokenIn).safeIncreaseAllowance(address(swapRouter), amountIn);

        // Prepare swap params
        ISwapRouter.ExactInputSingleParams memory params = ISwapRouter.ExactInputSingleParams({
            tokenIn: tokenIn,
            tokenOut: tokenOut,
            fee: fee,
            recipient: address(this),
            deadline: block.timestamp,
            amountIn: amountIn,
            amountOutMinimum: minAmountOut,
            sqrtPriceLimitX96: 0
        });

        // Execute swap
        amountOut = swapRouter.exactInputSingle(params);

        // Calculate actual slippage
        uint256 slippage = expectedOut > amountOut 
            ? ((expectedOut - amountOut) * 10000) / expectedOut
            : 0;

        emit Swapped(tokenIn, tokenOut, amountIn, amountOut, slippage);

        return amountOut;
    }

    /**
     * @notice Get quote for swap
     * @param tokenIn Input token address
     * @param tokenOut Output token address
     * @param amountIn Amount of input token
     * @param fee Pool fee tier
     * @return amountOut Expected amount of output tokens
     */
    function getQuote(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint24 fee
    ) public returns (uint256 amountOut) {
        return quoter.quoteExactInputSingle(
            tokenIn,
            tokenOut,
            fee,
            amountIn,
            0 // No price limit
        );
    }

    /**
     * @notice Multi-hop swap through path
     * @param path Encoded path (token0, fee0, token1, fee1, token2, ...)
     * @param amountIn Amount of input token
     * @param minAmountOut Minimum amount of output token
     * @return amountOut Actual amount of output tokens received
     */
    function swapMultiHop(
        bytes calldata path,
        uint256 amountIn,
        uint256 minAmountOut
    ) external onlyOwner returns (uint256 amountOut) {
        require(path.length >= 43, "Invalid path"); // At least 2 tokens + 1 fee
        require(amountIn > 0, "Amount must be > 0");

        // Extract first token from path
        address tokenIn = _toAddress(path, 0);

        // Approve router
        IERC20(tokenIn).safeIncreaseAllowance(address(swapRouter), amountIn);

        // Prepare swap params
        ISwapRouter.ExactInputParams memory params = ISwapRouter.ExactInputParams({
            path: path,
            recipient: address(this),
            deadline: block.timestamp,
            amountIn: amountIn,
            amountOutMinimum: minAmountOut
        });

        // Execute swap
        amountOut = swapRouter.exactInput(params);

        return amountOut;
    }

    /**
     * @notice Calculate optimal swap amount considering price impact
     * @param tokenIn Input token address
     * @param tokenOut Output token address
     * @param maxAmountIn Maximum amount willing to swap
     * @param fee Pool fee tier
     * @return optimalAmount Optimal amount to swap
     */
    function calculateOptimalSwapAmount(
        address tokenIn,
        address tokenOut,
        uint256 maxAmountIn,
        uint24 fee
    ) external returns (uint256 optimalAmount) {
        // Simple implementation: binary search for amount that doesn't exceed max slippage
        uint256 low = 0;
        uint256 high = maxAmountIn;
        
        while (high - low > maxAmountIn / 1000) { // Precision: 0.1%
            uint256 mid = (low + high) / 2;
            
            if (mid == 0) {
                return 0;
            }

            // Get quote for this amount
            uint256 expectedOut = getQuote(tokenIn, tokenOut, mid, fee);
            
            // Estimate slippage (simplified)
            uint256 pricePerToken = (expectedOut * 1e18) / mid;
            uint256 smallQuote = getQuote(tokenIn, tokenOut, mid / 100, fee);
            uint256 smallPricePerToken = (smallQuote * 1e18 * 100) / mid;
            
            if (smallPricePerToken == 0) {
                return 0;
            }

            uint256 priceImpact = smallPricePerToken > pricePerToken
                ? ((smallPricePerToken - pricePerToken) * 10000) / smallPricePerToken
                : 0;

            if (priceImpact <= maxSlippageBps) {
                low = mid;
            } else {
                high = mid;
            }
        }

        return low;
    }

    /**
     * @notice Update maximum slippage tolerance
     * @param slippageBps New slippage in basis points (100 = 1%)
     */
    function setMaxSlippage(uint256 slippageBps) external onlyOwner {
        require(slippageBps <= 1000, "Slippage too high"); // Max 10%
        uint256 oldSlippage = maxSlippageBps;
        maxSlippageBps = slippageBps;
        emit MaxSlippageUpdated(oldSlippage, slippageBps);
    }

    /**
     * @notice Internal: Convert bytes to address
     */
    function _toAddress(bytes memory data, uint256 start) internal pure returns (address) {
        require(data.length >= start + 20, "Invalid data");
        address addr;
        assembly {
            addr := mload(add(add(data, 0x20), start))
        }
        return addr;
    }

    /**
     * @notice Rescue stuck tokens
     * @dev For tokens accidentally sent to this contract
     */
    function rescueTokens(address token, address to, uint256 amount) external onlyOwner {
        require(to != address(0), "Invalid recipient");
        IERC20(token).safeTransfer(to, amount);
    }

    /**
     * @notice Rescue stuck ETH
     */
    function rescueETH(address payable to) external onlyOwner {
        require(to != address(0), "Invalid recipient");
        (bool success, ) = to.call{value: address(this).balance}("");
        require(success, "ETH transfer failed");
    }

    // Allow contract to receive ETH
    receive() external payable {}
}
