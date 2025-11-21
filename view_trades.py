"""
交易历史查询工具

用于查看交易记录和持仓历史的命令行工具
"""
import argparse
from decimal import Decimal
from termcolor import cprint
from tabulate import tabulate

from src.models import TradeHistoryManager


def format_decimal(value: Decimal, places: int = 2) -> str:
    """格式化 Decimal"""
    if value is None:
        return "N/A"
    return f"{float(value):.{places}f}"


def format_pnl(pnl: Decimal) -> str:
    """格式化盈亏（带颜色）"""
    if pnl is None:
        return "N/A"
    value = float(pnl)
    color = "green" if value > 0 else "red" if value < 0 else "white"
    return f"${value:+.2f}"


def view_open_positions(manager: TradeHistoryManager):
    """查看当前持仓"""
    positions = manager.get_all_open_positions()

    if not positions:
        cprint("\n✋ 当前无持仓", "yellow")
        return

    cprint(f"\n📊 当前持仓 ({len(positions)} 个)", "cyan")
    cprint("=" * 100, "cyan")

    for pos in positions:
        cprint(f"\n{pos.symbol} - {pos.side.upper()}", "white", attrs=['bold'])
        print(f"  持仓ID: {pos.position_id}")
        print(f"  开仓时间: {pos.entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  开仓价格: ${format_decimal(pos.entry_price)}")
        print(f"  数量: {format_decimal(pos.amount, 4)}")
        print(f"  杠杆: {pos.leverage}x")
        if pos.stop_loss:
            print(f"  止损: ${format_decimal(pos.stop_loss)}")
        else:
            cprint(f"  止损: ❌ 未设置", "red")
        if pos.take_profit:
            print(f"  止盈: ${format_decimal(pos.take_profit)}")
        else:
            cprint(f"  止盈: ❌ 未设置", "red")


def view_closed_positions(manager: TradeHistoryManager, days: int = None, limit: int = 10):
    """查看已平仓记录"""
    positions = manager.get_closed_positions(days=days)

    if not positions:
        cprint("\n✋ 暂无已平仓记录", "yellow")
        return

    # 限制显示数量
    if limit:
        positions = positions[:limit]

    cprint(f"\n📈 已平仓记录 (最近 {limit} 个)", "cyan")
    cprint("=" * 150, "cyan")

    table_data = []
    for pos in positions:
        duration = ""
        if pos.close_time:
            td = pos.close_time - pos.entry_time
            hours = td.total_seconds() / 3600
            duration = f"{hours:.1f}h"

        table_data.append([
            pos.symbol,
            pos.side.upper(),
            pos.entry_time.strftime('%m-%d %H:%M'),
            f"${format_decimal(pos.entry_price)}",
            f"${format_decimal(pos.close_price)}" if pos.close_price else "N/A",
            duration,
            format_pnl(pos.realized_pnl),
            f"{pos.realized_pnl_percent:+.1f}%" if pos.realized_pnl_percent else "N/A",
            pos.close_reason or "N/A",
        ])

    headers = ["交易对", "方向", "开仓时间", "开仓价", "平仓价", "持仓时长", "盈亏($)", "盈亏(%)", "平仓原因"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def view_trades(manager: TradeHistoryManager, days: int = None, limit: int = 20):
    """查看交易记录"""
    trades = manager.get_trades(days=days)

    if not trades:
        cprint("\n✋ 暂无交易记录", "yellow")
        return

    # 限制显示数量
    if limit:
        trades = trades[:limit]

    cprint(f"\n💱 交易记录 (最近 {limit} 笔)", "cyan")
    cprint("=" * 120, "cyan")

    table_data = []
    for trade in trades:
        table_data.append([
            trade.timestamp.strftime('%m-%d %H:%M:%S'),
            trade.symbol,
            trade.side.upper(),
            trade.action.upper(),
            f"${format_decimal(trade.price)}",
            format_decimal(trade.amount, 4),
            f"{trade.leverage}x" if trade.leverage else "N/A",
            f"${format_decimal(trade.fee, 4)}",
            format_pnl(trade.realized_pnl) if trade.realized_pnl else "N/A",
        ])

    headers = ["时间", "交易对", "方向", "动作", "价格", "数量", "杠杆", "手续费", "盈亏"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def view_statistics(manager: TradeHistoryManager, days: int = None):
    """查看统计数据"""
    stats = manager.get_statistics(days=days)

    period = f"最近 {days} 天" if days else "全部时间"
    cprint(f"\n📊 交易统计 ({period})", "cyan")
    cprint("=" * 60, "cyan")

    if stats['total_positions'] == 0:
        cprint("暂无交易数据", "yellow")
        return

    print(f"\n总持仓数: {stats['total_positions']}")
    print(f"盈利次数: {stats['win_count']}")
    print(f"亏损次数: {stats['loss_count']}")

    win_rate = stats['win_rate'] * 100
    win_rate_color = "green" if win_rate >= 50 else "red"
    cprint(f"胜率: {win_rate:.1f}%", win_rate_color)

    total_pnl = stats['total_pnl']
    pnl_color = "green" if total_pnl > 0 else "red"
    cprint(f"总盈亏: ${total_pnl:+.2f}", pnl_color)

    avg_pnl = stats['avg_pnl']
    avg_color = "green" if avg_pnl > 0 else "red"
    cprint(f"平均盈亏: ${avg_pnl:+.2f}", avg_color)

    print(f"最大盈利: ${stats['max_profit']:.2f}")
    print(f"最大亏损: ${stats['max_loss']:.2f}")
    print(f"平均持仓时长: {stats['avg_holding_time_hours']:.1f} 小时")


def main():
    parser = argparse.ArgumentParser(description="查看交易历史和持仓信息")
    parser.add_argument('--type', '-t', choices=['open', 'closed', 'trades', 'stats', 'all'],
                        default='all', help="查看类型")
    parser.add_argument('--days', '-d', type=int, help="查看最近N天的数据")
    parser.add_argument('--limit', '-l', type=int, default=20, help="限制显示数量")

    args = parser.parse_args()

    # 初始化管理器（使用统一数据库）
    manager = TradeHistoryManager(db_path="data/nofn.db")

    cprint("\n" + "=" * 60, "cyan")
    cprint("📊 交易历史查询工具", "cyan", attrs=['bold'])
    cprint("=" * 60, "cyan")

    if args.type in ['open', 'all']:
        view_open_positions(manager)

    if args.type in ['closed', 'all']:
        view_closed_positions(manager, days=args.days, limit=args.limit)

    if args.type in ['trades', 'all']:
        view_trades(manager, days=args.days, limit=args.limit)

    if args.type in ['stats', 'all']:
        view_statistics(manager, days=args.days)

    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 已退出")
    except Exception as e:
        cprint(f"\n❌ 错误: {e}", "red")
        import traceback
        traceback.print_exc()
