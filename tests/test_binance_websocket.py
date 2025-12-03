#!/usr/bin/env python3
"""
Binance WebSocket 独立测试脚本

使用方法:
    # 测试行情数据流
    python tests/test_binance_websocket.py --market

    # 测试用户数据流
    python tests/test_binance_websocket.py --user

    # 测试全部
    python tests/test_binance_websocket.py --all

    # 指定交易对
    python tests/test_binance_websocket.py --market --symbols BTCUSDC,ETHUSDC

    # 使用测试网
    python tests/test_binance_websocket.py --testnet --all
"""
import asyncio
import argparse
import sys
import os
from datetime import datetime
from decimal import Decimal

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from termcolor import cprint

from src.adapters.binance_websocket import (
    BinanceWebSocketManager,
    TickerData,
    KlineData,
    MarkPriceData,
)
from src.models import (
    OrderUpdateEvent,
    PositionUpdateEvent,
    TradeUpdateEvent,
    AccountUpdateEvent,
)
from src.utils.config_loader import load_config


class WebSocketTester:
    """WebSocket 测试器"""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        symbols: list = None,
    ):
        self.symbols = symbols or ["BTCUSDC", "ETHUSDC", "SOLUSDC"]
        self.ws_manager = BinanceWebSocketManager(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
        )

        # 统计
        self.ticker_count = 0
        self.kline_count = 0
        self.mark_price_count = 0
        self.order_count = 0
        self.trade_count = 0
        self.position_count = 0
        self.account_count = 0

    # ==================== 行情回调 ====================

    async def on_ticker(self, ticker: TickerData) -> None:
        """Ticker 回调"""
        self.ticker_count += 1
        if self.ticker_count <= 5 or self.ticker_count % 100 == 0:
            cprint(
                f"[Ticker #{self.ticker_count}] {ticker.symbol}: "
                f"${float(ticker.last_price):.2f} "
                f"({float(ticker.price_change_percent):+.2f}%)",
                "cyan"
            )

    async def on_kline(self, kline: KlineData) -> None:
        """K线回调"""
        self.kline_count += 1
        status = "✅ 已收盘" if kline.is_closed else "⏳ 进行中"
        if self.kline_count <= 5 or kline.is_closed:
            cprint(
                f"[K线 #{self.kline_count}] {kline.symbol} {kline.interval}: "
                f"O={float(kline.open):.2f} H={float(kline.high):.2f} "
                f"L={float(kline.low):.2f} C={float(kline.close):.2f} {status}",
                "green" if kline.is_closed else "yellow"
            )

    async def on_mark_price(self, mark_price: MarkPriceData) -> None:
        """标记价格回调"""
        self.mark_price_count += 1
        if self.mark_price_count <= 5 or self.mark_price_count % 50 == 0:
            cprint(
                f"[标记价格 #{self.mark_price_count}] {mark_price.symbol}: "
                f"Mark=${float(mark_price.mark_price):.2f} "
                f"Index=${float(mark_price.index_price):.2f} "
                f"FR={float(mark_price.funding_rate) * 100:.4f}%",
                "magenta"
            )

    # ==================== 用户数据回调 ====================

    async def on_order_update(self, order: OrderUpdateEvent) -> None:
        """订单更新回调"""
        self.order_count += 1
        status_emoji = "✅" if order.is_filled() else "⏳"
        sl_tp = ""
        if order.is_stop_loss:
            sl_tp = " [止损]"
        elif order.is_take_profit:
            sl_tp = " [止盈]"

        cprint(
            f"\n{status_emoji} [订单 #{self.order_count}] {order.symbol}{sl_tp}\n"
            f"   ID: {order.order_id}\n"
            f"   类型: {order.order_type}\n"
            f"   状态: {order.status}\n"
            f"   方向: {order.side}\n"
            f"   数量: {order.amount} (已成交: {order.filled})",
            "green" if order.is_filled() else "yellow"
        )

    async def on_trade_update(self, trade: TradeUpdateEvent) -> None:
        """成交更新回调"""
        self.trade_count += 1
        pnl_info = ""
        if trade.is_closing and trade.realized_pnl:
            pnl_emoji = "🟢" if trade.realized_pnl > 0 else "🔴"
            pnl_info = f"\n   {pnl_emoji} 已实现盈亏: ${float(trade.realized_pnl):.2f}"

        cprint(
            f"\n💰 [成交 #{self.trade_count}] {trade.symbol}\n"
            f"   价格: ${float(trade.price):.2f}\n"
            f"   数量: {trade.amount}\n"
            f"   手续费: {trade.fee} {trade.fee_currency}"
            f"{pnl_info}",
            "cyan"
        )

    async def on_position_update(self, position: PositionUpdateEvent) -> None:
        """仓位更新回调"""
        self.position_count += 1

        if position.is_closed:
            reason = position.get_close_reason_text()
            cprint(
                f"\n⚠️  [仓位 #{self.position_count}] {position.symbol} 已平仓\n"
                f"   原持仓: {position.previous_amount}\n"
                f"   平仓原因: {reason}",
                "yellow"
            )
        else:
            change_text = f"+{position.amount_change}" if position.amount_change > 0 else str(position.amount_change)
            cprint(
                f"\n📊 [仓位 #{self.position_count}] {position.symbol}\n"
                f"   方向: {position.side}\n"
                f"   当前持仓: {position.position_amount}\n"
                f"   变化: {change_text}\n"
                f"   入场价: ${float(position.entry_price):.2f}\n"
                f"   未实现盈亏: ${float(position.unrealized_pnl):.2f}",
                "cyan"
            )

    async def on_account_update(self, account: AccountUpdateEvent) -> None:
        """账户更新回调"""
        self.account_count += 1
        cprint(
            f"\n💳 [账户 #{self.account_count}]\n"
            f"   总余额: ${float(account.total_balance):.2f}\n"
            f"   可用余额: ${float(account.available_balance):.2f}",
            "blue"
        )

    # ==================== 测试方法 ====================

    async def test_market_stream(self, duration: int = 60) -> None:
        """测试行情数据流"""
        cprint("\n" + "=" * 60, "white")
        cprint("  📈 行情数据流测试", "white", attrs=["bold"])
        cprint("=" * 60, "white")
        cprint(f"交易对: {', '.join(self.symbols)}", "white")
        cprint(f"测试时长: {duration} 秒", "white")
        cprint("=" * 60 + "\n", "white")

        # 订阅行情流
        await self.ws_manager.subscribe_market_streams(
            symbols=self.symbols,
            streams=["miniTicker", "markPrice", "kline_1m"],
            on_ticker=self.on_ticker,
            on_kline=self.on_kline,
            on_mark_price=self.on_mark_price,
        )

        # 运行指定时长
        try:
            await asyncio.sleep(duration)
        except asyncio.CancelledError:
            pass

        # 打印统计
        cprint("\n" + "=" * 60, "white")
        cprint("  📊 行情流测试统计", "white", attrs=["bold"])
        cprint("=" * 60, "white")
        cprint(f"Ticker 消息: {self.ticker_count}", "cyan")
        cprint(f"K线 消息: {self.kline_count}", "green")
        cprint(f"标记价格 消息: {self.mark_price_count}", "magenta")
        cprint("=" * 60 + "\n", "white")

    async def test_user_stream(self, duration: int = 300) -> None:
        """测试用户数据流"""
        cprint("\n" + "=" * 60, "white")
        cprint("  👤 用户数据流测试", "white", attrs=["bold"])
        cprint("=" * 60, "white")
        cprint(f"测试时长: {duration} 秒", "white")
        cprint("提示: 在测试期间进行交易操作以查看实时更新", "yellow")
        cprint("=" * 60 + "\n", "white")

        # 订阅用户数据流
        success = await self.ws_manager.subscribe_user_stream(
            on_order_update=self.on_order_update,
            on_trade_update=self.on_trade_update,
            on_position_update=self.on_position_update,
            on_account_update=self.on_account_update,
        )

        if not success:
            cprint("❌ 用户数据流订阅失败", "red")
            return

        # 运行指定时长
        try:
            await asyncio.sleep(duration)
        except asyncio.CancelledError:
            pass

        # 打印统计
        cprint("\n" + "=" * 60, "white")
        cprint("  📊 用户数据流测试统计", "white", attrs=["bold"])
        cprint("=" * 60, "white")
        cprint(f"订单更新: {self.order_count}", "green")
        cprint(f"成交更新: {self.trade_count}", "cyan")
        cprint(f"仓位更新: {self.position_count}", "yellow")
        cprint(f"账户更新: {self.account_count}", "blue")
        cprint("=" * 60 + "\n", "white")

    async def test_all(self, duration: int = 300) -> None:
        """测试全部数据流"""
        cprint("\n" + "=" * 60, "white")
        cprint("  🚀 全部数据流测试", "white", attrs=["bold"])
        cprint("=" * 60, "white")
        cprint(f"交易对: {', '.join(self.symbols)}", "white")
        cprint(f"测试时长: {duration} 秒", "white")
        cprint("=" * 60 + "\n", "white")

        # 订阅行情流
        await self.ws_manager.subscribe_market_streams(
            symbols=self.symbols,
            streams=["miniTicker", "markPrice"],
            on_ticker=self.on_ticker,
            on_mark_price=self.on_mark_price,
        )

        # 订阅用户数据流
        await self.ws_manager.subscribe_user_stream(
            on_order_update=self.on_order_update,
            on_trade_update=self.on_trade_update,
            on_position_update=self.on_position_update,
            on_account_update=self.on_account_update,
        )

        # 运行指定时长
        try:
            await asyncio.sleep(duration)
        except asyncio.CancelledError:
            pass

        # 打印统计
        cprint("\n" + "=" * 60, "white")
        cprint("  📊 测试统计", "white", attrs=["bold"])
        cprint("=" * 60, "white")
        cprint(f"Ticker: {self.ticker_count}", "cyan")
        cprint(f"标记价格: {self.mark_price_count}", "magenta")
        cprint(f"订单: {self.order_count}", "green")
        cprint(f"成交: {self.trade_count}", "cyan")
        cprint(f"仓位: {self.position_count}", "yellow")
        cprint(f"账户: {self.account_count}", "blue")
        cprint("=" * 60 + "\n", "white")

    async def cleanup(self) -> None:
        """清理资源"""
        await self.ws_manager.stop()


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Binance WebSocket 测试")
    parser.add_argument("--market", action="store_true", help="测试行情数据流")
    parser.add_argument("--user", action="store_true", help="测试用户数据流")
    parser.add_argument("--all", action="store_true", help="测试全部数据流")
    parser.add_argument("--symbols", type=str, default="BTCUSDC,ETHUSDC,SOLUSDC",
                        help="交易对列表，逗号分隔 (默认: BTCUSDC,ETHUSDC,SOLUSDC)")
    parser.add_argument("--duration", type=int, default=60, help="测试时长(秒) (默认: 60)")
    parser.add_argument("--testnet", action="store_true", help="使用测试网")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")

    args = parser.parse_args()

    # 至少选择一种测试
    if not (args.market or args.user or args.all):
        parser.print_help()
        print("\n请至少选择一种测试模式: --market, --user, 或 --all")
        return

    # 加载配置
    try:
        config = load_config(args.config)
        api_key = config.exchange.api_key
        api_secret = config.exchange.api_secret
    except Exception as e:
        cprint(f"❌ 加载配置失败: {e}", "red")
        cprint("请确保 config/config.yaml 存在且包含 exchange.api_key 和 exchange.api_secret", "yellow")
        return

    # 解析交易对
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # 创建测试器
    tester = WebSocketTester(
        api_key=api_key,
        api_secret=api_secret,
        testnet=args.testnet,
        symbols=symbols,
    )

    try:
        # 运行测试
        if args.all:
            await tester.test_all(duration=args.duration)
        elif args.market:
            await tester.test_market_stream(duration=args.duration)
        elif args.user:
            await tester.test_user_stream(duration=args.duration)

    except KeyboardInterrupt:
        cprint("\n⚠️  用户中断测试", "yellow")
    finally:
        await tester.cleanup()
        cprint("✅ 测试完成", "green")


if __name__ == "__main__":
    asyncio.run(main())