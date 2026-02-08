"""
Pre-built Dune Analytics Queries for Aave V3 and Curve Finance

These queries can be saved to Dune.com and referenced by ID,
or executed directly via API
"""


class AaveV3Queries:
    """SQL queries for Aave V3 protocol data"""
    
    @staticmethod
    def get_market_rates() -> str:
        """
        Get current supply/borrow rates for all Aave V3 markets
        Returns: asset, supply_apy, variable_borrow_apy, total_liquidity_usd, utilization_rate
        """
        return """
        WITH latest_reserves AS (
            SELECT 
                reserve,
                asset_symbol,
                liquidity_rate / 1e27 * 100 as supply_apy,
                variable_borrow_rate / 1e27 * 100 as variable_borrow_apy,
                stable_borrow_rate / 1e27 * 100 as stable_borrow_apy,
                available_liquidity / POWER(10, decimals) as available_liquidity,
                total_stable_debt / POWER(10, decimals) as total_stable_debt,
                total_variable_debt / POWER(10, decimals) as total_variable_debt,
                decimals,
                evt_block_time,
                evt_block_number
            FROM aave_v3_ethereum.Pool_evt_ReserveDataUpdated r
            WHERE evt_block_time >= NOW() - INTERVAL '24' HOUR
            ORDER BY evt_block_time DESC
        ),
        ranked_reserves AS (
            SELECT 
                *,
                ROW_NUMBER() OVER (PARTITION BY reserve ORDER BY evt_block_time DESC) as rn
            FROM latest_reserves
        ),
        current_reserves AS (
            SELECT 
                reserve,
                asset_symbol,
                supply_apy,
                variable_borrow_apy,
                stable_borrow_apy,
                available_liquidity,
                total_stable_debt,
                total_variable_debt,
                available_liquidity + total_stable_debt + total_variable_debt as total_liquidity,
                CASE 
                    WHEN (available_liquidity + total_stable_debt + total_variable_debt) > 0
                    THEN (total_stable_debt + total_variable_debt) / (available_liquidity + total_stable_debt + total_variable_debt) * 100
                    ELSE 0
                END as utilization_rate,
                evt_block_time
            FROM ranked_reserves
            WHERE rn = 1
        )
        SELECT 
            asset_symbol,
            reserve as asset_address,
            supply_apy,
            variable_borrow_apy,
            stable_borrow_apy,
            total_liquidity,
            available_liquidity,
            utilization_rate,
            evt_block_time as last_update
        FROM current_reserves
        WHERE supply_apy > 0
        ORDER BY total_liquidity DESC
        LIMIT 50
        """
    
    @staticmethod
    def get_protocol_tvl() -> str:
        """Get total TVL across all Aave V3 markets"""
        return """
        WITH latest_block AS (
            SELECT MAX(evt_block_number) as block_num
            FROM aave_v3_ethereum.Pool_evt_ReserveDataUpdated
            WHERE evt_block_time >= NOW() - INTERVAL '1' HOUR
        )
        SELECT 
            COUNT(DISTINCT reserve) as market_count,
            SUM(available_liquidity / POWER(10, decimals)) as total_available_liquidity,
            SUM((total_stable_debt + total_variable_debt) / POWER(10, decimals)) as total_debt
        FROM aave_v3_ethereum.Pool_evt_ReserveDataUpdated r
        CROSS JOIN latest_block
        WHERE evt_block_number = latest_block.block_num
        """


class CurveFinanceQueries:
    """SQL queries for Curve Finance protocol data"""
    
    @staticmethod
    def get_pool_apys() -> str:
        """
        Get current APYs for major Curve pools
        Returns: pool_name, pool_address, base_apy, tvl_usd
        """
        return """
        WITH latest_swaps AS (
            SELECT 
                contract_address as pool_address,
                MAX(block_time) as last_swap_time,
                COUNT(*) as swap_count_24h
            FROM curvefi_ethereum.CurvePool_evt_TokenExchange
            WHERE block_time >= NOW() - INTERVAL '24' HOUR
            GROUP BY contract_address
        ),
        pool_volumes AS (
            SELECT 
                s.contract_address as pool_address,
                SUM(
                    CASE 
                        WHEN tokens_bought > 0 THEN tokens_bought / POWER(10, 18)
                        ELSE 0
                    END
                ) as volume_24h
            FROM curvefi_ethereum.CurvePool_evt_TokenExchange s
            WHERE s.block_time >= NOW() - INTERVAL '24' HOUR
            GROUP BY s.contract_address
        )
        SELECT 
            p.pool_address,
            ls.swap_count_24h,
            pv.volume_24h,
            ls.last_swap_time
        FROM latest_swaps ls
        LEFT JOIN pool_volumes pv ON ls.pool_address = pv.pool_address
        LEFT JOIN latest_swaps p ON ls.pool_address = p.pool_address
        WHERE ls.swap_count_24h > 10
        ORDER BY ls.swap_count_24h DESC
        LIMIT 30
        """
    
    @staticmethod
    def get_stable_pools() -> str:
        """Get Curve stablecoin pool data (3pool, FRAX, etc)"""
        return """
        SELECT 
            contract_address as pool_address,
            COUNT(*) as daily_swaps,
            SUM(tokens_bought) / POWER(10, 18) as daily_volume
        FROM curvefi_ethereum.CurvePool_evt_TokenExchange
        WHERE block_time >= NOW() - INTERVAL '24' HOUR
        GROUP BY contract_address
        HAVING COUNT(*) > 50
        ORDER BY daily_volume DESC
        LIMIT 20
        """


# Query IDs for saved queries on Dune (to be created)
DUNE_QUERY_IDS = {
    "aave_v3_rates": None,  # Create at dune.com and add ID here
    "aave_v3_tvl": None,
    "curve_pool_apys": None,
    "curve_stable_pools": None,
}
