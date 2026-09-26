"""CLI flag defaults for the secondary Alpaca bridge (no network, no credentials)."""

from __future__ import annotations

import pytest

pytest.importorskip("dotenv")
pytest.importorskip("requests")

import live_trading  # noqa: E402


def test_defaults_are_paper_and_dry_run() -> None:
    args = live_trading.build_parser().parse_args([])
    assert args.paper is True
    assert args.dry_run is True
    assert args.live is False


def test_no_paper_selects_live_account_without_enabling_live_orders() -> None:
    args = live_trading.build_parser().parse_args(["--no-paper"])
    assert args.paper is False
    assert args.live is False
    assert args.dry_run is True


def test_live_flag_still_required_for_order_submission() -> None:
    args = live_trading.build_parser().parse_args(["--live", "--paper"])
    assert args.live is True
    assert args.paper is True
