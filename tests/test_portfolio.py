from stat_arb_engine.execution import CostModel, ExecutionHandler, LimitOrderBook
from stat_arb_engine.portfolio import PortfolioLedger


def make_fill(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    commission: float = 0.0,
    execution_cost: float = 0.0,
) -> dict:
    signed_qty = qty if side == "buy" else -qty
    gross_cash_flow = -signed_qty * price
    return {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": price,
        "commission": commission,
        "execution_cost": execution_cost,
        "net_cash_flow": gross_cash_flow - commission - execution_cost,
    }



def test_long_unrealized_pnl_and_equity() -> None:
    ledger = PortfolioLedger(starting_cash=10_000.0)
    ledger.process_fill(make_fill("A", "buy", 100, 100.0))
    snapshot = ledger.mark_to_market({"A": 101.0})
    assert snapshot["unrealized_pnl"] == 100.0
    assert snapshot["equity"] == 10_100.0



def test_short_unrealized_pnl_and_equity() -> None:
    ledger = PortfolioLedger(starting_cash=10_000.0)
    ledger.process_fill(make_fill("A", "sell", 100, 100.0))
    snapshot = ledger.mark_to_market({"A": 99.0})
    assert snapshot["unrealized_pnl"] == 100.0
    assert snapshot["equity"] == 10_100.0



def test_round_trip_costs_reduce_flat_equity() -> None:
    ledger = PortfolioLedger(starting_cash=10_000.0)
    ledger.process_fill(
        make_fill("A", "buy", 100, 100.0, commission=1.0, execution_cost=4.0)
    )
    ledger.process_fill(
        make_fill("A", "sell", 100, 100.0, commission=1.0, execution_cost=4.0)
    )
    snapshot = ledger.mark_to_market({"A": 100.0})
    assert snapshot["equity"] == 9_990.0
    assert snapshot["realized_pnl"] == 0.0
    assert snapshot["commissions"] == 2.0
    assert snapshot["transaction_costs"] == 8.0



def test_flat_equity_reconciles_realized_and_costs() -> None:
    ledger = PortfolioLedger(starting_cash=10_000.0)
    ledger.process_fill(
        make_fill("A", "buy", 100, 100.0, commission=1.0, execution_cost=4.0)
    )
    ledger.process_fill(
        make_fill("A", "sell", 100, 102.0, commission=1.0, execution_cost=4.0)
    )
    snapshot = ledger.mark_to_market({"A": 102.0})
    assert snapshot["equity"] == 10_190.0
    assert snapshot["realized_pnl"] == 200.0
    assert snapshot["equity"] - ledger.starting_cash == (
        snapshot["realized_pnl"]
        - snapshot["commissions"]
        - snapshot["transaction_costs"]
    )


def test_partial_close_tracks_cost_basis_and_net_pnl_invariant() -> None:
    ledger = PortfolioLedger(starting_cash=10_000.0)
    ledger.process_fill(make_fill("A", "buy", 100, 100.0))
    ledger.process_fill(make_fill("A", "sell", 40, 110.0))
    snapshot = ledger.mark_to_market({"A": 111.0})

    assert snapshot["positions"]["A"]["quantity"] == 60.0
    assert snapshot["positions"]["A"]["average_cost"] == 100.0
    assert snapshot["positions"]["A"]["cost_basis"] == 6_000.0
    assert snapshot["positions"]["A"]["market_value"] == 6_660.0
    assert snapshot["realized_pnl"] == 400.0
    assert snapshot["unrealized_pnl"] == 660.0
    assert snapshot["gross_pnl"] == 1_060.0
    assert snapshot["net_pnl"] == 1_060.0
    assert snapshot["equity"] - ledger.starting_cash == snapshot["net_pnl"]


def test_position_reversal_resets_average_cost_for_new_short() -> None:
    ledger = PortfolioLedger(starting_cash=10_000.0)
    ledger.process_fill(make_fill("A", "buy", 100, 100.0))
    ledger.process_fill(make_fill("A", "sell", 150, 102.0))
    snapshot = ledger.mark_to_market({"A": 101.0})

    assert snapshot["positions"]["A"]["quantity"] == -50.0
    assert snapshot["positions"]["A"]["average_cost"] == 102.0
    assert snapshot["positions"]["A"]["cost_basis"] == -5_100.0
    assert snapshot["realized_pnl"] == 200.0
    assert snapshot["unrealized_pnl"] == 50.0
    assert snapshot["equity"] - ledger.starting_cash == snapshot["net_pnl"]


def test_positive_costs_cannot_improve_net_cash_flow() -> None:
    lob = LimitOrderBook()
    lob.update(99.99, 100.01, depth_qty=500)
    execution = ExecutionHandler(
        lob,
        CostModel(spread_bps=5.0, slippage_bps=5.0, impact_bps=5.0),
    )
    fill = execution.submit_order(
        {
            "type": "MARKET",
            "symbol": "A",
            "side": "buy",
            "qty": 100,
            "timestamp": "2024-01-01",
        }
    )
    assert fill is not None
    assert fill["total_cost"] > 0.0
    assert fill["net_cash_flow"] < fill["gross_cash_flow"]
