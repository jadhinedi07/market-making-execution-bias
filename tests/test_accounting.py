"""
Hand-calculated accounting example.

Mid path:  t=0 mid 100.5 | t=3 mid 101.5 | t=5 mid 101.0 (final)
Fills:     t=1 BUY  2 @ 100 (mid 100.5)
           t=2 SELL 1 @ 101 (mid 100.5)
           t=4 SELL 2 @ 102 (mid 101.5)

cash      = -200 + 101 + 204                        = 105
position  = 2 - 1 - 2                               = -1
gross PnL = cash + position * final mid = 105 - 101 = 4

spread capture = (+1)(100.5-100)(2) + (-1)(100.5-101)(1) + (-1)(101.5-102)(2)
               = 1 + 0.5 + 1                        = 2.5
inventory PnL  = pos(t=3) * (101.5-100.5) + pos(t=5) * (101.0-101.5)
               = (+1)(1.0) + (-1)(-0.5)             = 1.5
check: 2.5 + 1.5 = 4 = gross

Average cost: buy 2 @100 -> avg 100. Sell 1 @101 -> realised +1.
Sell 2 @102 closes 1 more (+2, realised 3), flips to -1 with avg 102.
unrealised = -1 * (101 - 102) = +1. realised + unrealised = 4.

Fees at 2 bp on notional 200 + 101 + 204 = 505 -> 0.101.
"""
import numpy as np
import pytest

from mmbias.accounting import (cash_and_position, equity_curve, fills_to_arrays, inventory_pnl,
                               markouts, max_drawdown, realised_unrealised, spread_capture,
                               weighted_mean)

FILLS = [(1, +1, 100.0, 2.0, 100.5, False, 0.0),
         (2, -1, 101.0, 1.0, 100.5, False, 0.0),
         (4, -1, 102.0, 2.0, 101.5, False, 0.0)]
MID_T = np.array([0, 3, 5])
MID_V = np.array([100.5, 101.5, 101.0])


def test_cash_position_fees():
    f = fills_to_arrays(FILLS)
    cash, pos, fees = cash_and_position(f, fee_bps=2.0)
    assert cash == pytest.approx(105.0)
    assert pos == pytest.approx(-1.0)
    assert fees == pytest.approx(0.101)


def test_decomposition_identity():
    f = fills_to_arrays(FILLS)
    cash, pos, _ = cash_and_position(f)
    gross = cash + pos * MID_V[-1]
    assert gross == pytest.approx(4.0)
    assert spread_capture(f) == pytest.approx(2.5)
    assert inventory_pnl(f, MID_T, MID_V) == pytest.approx(1.5)
    assert spread_capture(f) + inventory_pnl(f, MID_T, MID_V) == pytest.approx(gross)


def test_realised_unrealised_average_cost():
    f = fills_to_arrays(FILLS)
    r, u = realised_unrealised(f, mark=101.0)
    assert r == pytest.approx(3.0)
    assert u == pytest.approx(1.0)


def test_equity_curve_and_drawdown():
    f = fills_to_arrays(FILLS)
    # equity at t=0: 0. t=3: cash -200+101 = -99, pos 1, mid 101.5 -> 2.5.
    # t=5: cash 105, pos -1, mid 101 -> 4.0
    _, eq = equity_curve(f, MID_T, MID_V)
    assert eq == pytest.approx([0.0, 2.5, 4.0])
    assert max_drawdown(np.array([0.0, 3.0, 1.0, 4.0, 0.5])) == pytest.approx(-3.5)


def test_markouts_hand_example():
    f = fills_to_arrays(FILLS)
    mk = markouts(f, MID_T, MID_V, horizon_us=2, end_ts=5)
    # fill 1 (bid, t=1): mid at t=3 is 101.5 -> +1 * (101.5 - 100.5)/100.5 bps
    # fill 2 (ask, t=2): mid at t=4 is 101.5 -> -1 * (101.5 - 100.5)/100.5 bps (adverse)
    # fill 3 (ask, t=4): t+2 = 6 is past the end -> NaN
    assert mk[0] == pytest.approx(1e4 * 1.0 / 100.5)
    assert mk[1] == pytest.approx(-1e4 * 1.0 / 100.5)
    assert np.isnan(mk[2])
    assert weighted_mean(mk, f["qty"]) == pytest.approx((2 * mk[0] + 1 * mk[1]) / 3)
