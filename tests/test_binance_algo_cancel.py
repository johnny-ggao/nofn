"""测试 Binance Futures Algo Orders 撤单功能。

这个脚本测试：
1. 创建止损止盈条件订单
2. 查询 Algo Orders
3. 撤销 Algo Orders

使用前需要设置环境变量:
- BINANCE_API_KEY
- BINANCE_SECRET_KEY
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import ccxt.async_support as ccxt
from termcolor import cprint


async def test_algo_order_cancel():
    """完整测试 Algo Orders 撤单流程。"""

    api_key = os.environ.get("BINANCE_API_KEY", "")
    secret = os.environ.get("BINANCE_SECRET_KEY", "") or os.environ.get("BINANCE_SECRET", "")

    if not api_key or not secret:
        cprint("请设置 BINANCE_API_KEY 和 BINANCE_SECRET_KEY 环境变量", "red")
        return

    # 创建 exchange 实例 - 使用真实网络
    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",
        },
    })

    try:
        await exchange.load_markets()
        cprint("Markets loaded", "green")

        symbol = "BTC/USDC:USDC"
        binance_symbol = "BTCUSDC"

        # Step 1: 查询当前持仓
        cprint("\n=== Step 1: 查询持仓 ===", "yellow")
        positions = await exchange.fetch_positions([symbol])
        position = None
        for p in positions:
            if p.get("symbol") == symbol and abs(float(p.get("contracts", 0))) > 0:
                position = p
                break

        if position:
            cprint(f"当前持仓: {position.get('side')} {position.get('contracts')} @ {position.get('entryPrice')}", "cyan")
        else:
            cprint("没有持仓", "yellow")

        # Step 2: 获取当前价格
        cprint("\n=== Step 2: 获取价格 ===", "yellow")
        ticker = await exchange.fetch_ticker(symbol)
        current_price = float(ticker.get("last", 0))
        cprint(f"当前价格: {current_price}", "cyan")

        # Step 3: 查询并测试撤销 Algo Orders
        cprint("\n=== Step 3: 查询 Algo Orders ===", "yellow")
        try:
            algo_orders = await exchange.fapiPrivateGetOpenAlgoOrders({
                "symbol": binance_symbol,
            })
            orders_list = algo_orders.get("orders", []) if isinstance(algo_orders, dict) else algo_orders
            cprint(f"当前 Algo Orders 数量: {len(orders_list)}", "cyan")
            for o in orders_list:
                cprint(f"  - algoId={o.get('algoId')}, type={o.get('algoType')}, status={o.get('algoStatus')}, side={o.get('side')}", "white")
        except Exception as e:
            cprint(f"查询 Algo Orders 失败: {e}", "red")
            orders_list = []

        # Step 4: 如果有 Algo Orders，尝试撤销
        if orders_list:
            cprint("\n=== Step 4: 撤销 Algo Orders ===", "yellow")

            # 方法1: 批量撤销
            cprint("尝试批量撤销...", "magenta")
            try:
                result = await exchange.fapiPrivateDeleteAlgoOpenOrders({
                    "symbol": binance_symbol,
                })
                cprint(f"批量撤销结果: {result}", "green")
            except Exception as e:
                cprint(f"批量撤销失败: {e}", "red")

                # 方法2: 单个撤销
                cprint("尝试单个撤销...", "magenta")
                for order in orders_list:
                    algo_id = order.get("algoId")
                    if algo_id:
                        try:
                            result = await exchange.fapiPrivateDeleteAlgoOrder({
                                "symbol": binance_symbol,
                                "algoId": algo_id,
                            })
                            cprint(f"撤销 {algo_id} 成功: {result}", "green")
                        except Exception as e:
                            cprint(f"撤销 {algo_id} 失败: {e}", "red")

            # 验证撤销结果
            await asyncio.sleep(1)
            cprint("\n=== 验证撤销结果 ===", "yellow")
            algo_orders = await exchange.fapiPrivateGetOpenAlgoOrders({
                "symbol": binance_symbol,
            })
            orders_list = algo_orders.get("orders", []) if isinstance(algo_orders, dict) else algo_orders
            cprint(f"撤销后 Algo Orders 数量: {len(orders_list)}", "cyan")
            if len(orders_list) == 0:
                cprint("所有 Algo Orders 已成功撤销!", "green")
            else:
                cprint("还有未撤销的订单!", "red")
                for o in orders_list:
                    cprint(f"  - algoId={o.get('algoId')}, type={o.get('algoType')}, status={o.get('algoStatus')}", "white")
        else:
            cprint("\n没有需要撤销的 Algo Orders", "yellow")

            # 可选: 创建测试订单
            cprint("\n是否要创建测试订单? (需要有持仓)", "yellow")

    except Exception as e:
        cprint(f"测试失败: {e}", "red")
        import traceback
        traceback.print_exc()
    finally:
        await exchange.close()


async def test_full_flow():
    """完整流程测试：开仓 -> 下止损止盈 -> 平仓 -> 验证撤单。

    注意: 这个测试会在真实账户上执行交易，请谨慎使用!
    """

    api_key = os.environ.get("BINANCE_API_KEY", "")
    secret = os.environ.get("BINANCE_SECRET_KEY", "") or os.environ.get("BINANCE_SECRET", "")

    if not api_key or not secret:
        cprint("请设置 BINANCE_API_KEY 和 BINANCE_SECRET_KEY 环境变量", "red")
        return

    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",
        },
    })

    try:
        await exchange.load_markets()
        cprint("Markets loaded", "green")

        symbol = "BTC/USDC:USDC"
        binance_symbol = "BTCUSDC"

        # 获取当前价格
        ticker = await exchange.fetch_ticker(symbol)
        current_price = float(ticker.get("last", 0))
        cprint(f"当前价格: {current_price}", "cyan")

        # Step 1: 开多仓
        cprint("\n=== Step 1: 开多仓 ===", "yellow")
        order = await exchange.create_order(
            symbol=symbol,
            type="market",
            side="buy",
            amount=0.001,
            params={"reduceOnly": False},
        )
        cprint(f"开仓订单: {order.get('id')}, status={order.get('status')}, filled={order.get('filled')}", "green")
        await asyncio.sleep(1)

        # Step 2: 创建止损止盈订单
        cprint("\n=== Step 2: 创建止损止盈订单 ===", "yellow")
        sl_price = round(current_price * 0.95, 1)  # 5% 止损
        tp_price = round(current_price * 1.05, 1)  # 5% 止盈
        cprint(f"止损价: {sl_price}, 止盈价: {tp_price}", "cyan")

        # 止损
        try:
            sl_order = await exchange.create_order(
                symbol=symbol,
                type="STOP_MARKET",
                side="sell",
                amount=0.001,
                price=None,
                params={
                    "stopPrice": sl_price,
                    "reduceOnly": True,
                    "workingType": "MARK_PRICE",
                },
            )
            cprint(f"止损订单: {sl_order}", "green")
        except Exception as e:
            cprint(f"止损订单失败: {e}", "red")

        # 止盈
        try:
            tp_order = await exchange.create_order(
                symbol=symbol,
                type="TAKE_PROFIT_MARKET",
                side="sell",
                amount=0.001,
                price=None,
                params={
                    "stopPrice": tp_price,
                    "reduceOnly": True,
                    "workingType": "MARK_PRICE",
                },
            )
            cprint(f"止盈订单: {tp_order}", "green")
        except Exception as e:
            cprint(f"止盈订单失败: {e}", "red")

        await asyncio.sleep(1)

        # Step 3: 查询 Algo Orders
        cprint("\n=== Step 3: 查询 Algo Orders ===", "yellow")
        algo_orders = await exchange.fapiPrivateGetOpenAlgoOrders({
            "symbol": binance_symbol,
        })
        orders_list = algo_orders.get("orders", []) if isinstance(algo_orders, dict) else algo_orders
        cprint(f"Algo Orders 数量: {len(orders_list)}", "cyan")
        for o in orders_list:
            cprint(f"  - algoId={o.get('algoId')}, type={o.get('algoType')}, side={o.get('side')}, stopPrice={o.get('stopPrice')}", "white")

        # Step 4: 平仓
        cprint("\n=== Step 4: 平仓 ===", "yellow")
        close_order = await exchange.create_order(
            symbol=symbol,
            type="market",
            side="sell",
            amount=0.001,
            params={"reduceOnly": True},
        )
        cprint(f"平仓订单: {close_order.get('id')}, status={close_order.get('status')}", "green")
        await asyncio.sleep(1)

        # Step 5: 撤销 Algo Orders
        cprint("\n=== Step 5: 撤销 Algo Orders ===", "yellow")
        try:
            result = await exchange.fapiPrivateDeleteAlgoOpenOrders({
                "symbol": binance_symbol,
            })
            cprint(f"批量撤销结果: {result}", "green")
        except Exception as e:
            cprint(f"批量撤销失败: {e}", "red")

        # Step 6: 验证
        cprint("\n=== Step 6: 验证 ===", "yellow")
        await asyncio.sleep(1)
        algo_orders = await exchange.fapiPrivateGetOpenAlgoOrders({
            "symbol": binance_symbol,
        })
        orders_list = algo_orders.get("orders", []) if isinstance(algo_orders, dict) else algo_orders
        if len(orders_list) == 0:
            cprint("成功! 所有 Algo Orders 已撤销", "green")
        else:
            cprint(f"失败! 还有 {len(orders_list)} 个 Algo Orders", "red")

    except Exception as e:
        cprint(f"测试失败: {e}", "red")
        import traceback
        traceback.print_exc()
    finally:
        await exchange.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        cprint("运行完整流程测试 (会产生真实交易!)", "red")
        asyncio.run(test_full_flow())
    else:
        cprint("运行查询/撤销测试", "cyan")
        asyncio.run(test_algo_order_cancel())
