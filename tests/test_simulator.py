"""
Hand-traced scenarios through the full simulator (no files, events fed directly).

Times are in microseconds. Latency is 10 us, so a decision at t goes live at t + 10.
Order size q = 2, inventory limit 10.
"""
import pytest

from mmbias.simulate import AgentSpec, Simulator

T, F = True, False


def make(lat=10):
    specs = [AgentSpec("A", "A", latency_us=lat), AgentSpec("B", "B", "prop", lat),
             AgentSpec("C", "C", latency_us=lat)]
    sim = Simulator(specs, q=2.0, qmax=10.0)
    return sim, {a.name: a for a in sim.agents}


def snapshot(sim, t):
    # bids 100 x5, 99 x4 | asks 101 x5, 102 x4
    sim.on_message(t, [T, T, T, T], [T, T, F, F], [100.0, 99.0, 101.0, 102.0], [5, 4, 5, 4])


def test_queue_scenario_all_three_models():
    sim, ag = make()
    snapshot(sim, 0)                     # every agent decides: bid 100, ask 101, live at t=10
    sim.on_trade(5, 100.0, 3, -1)        # BEFORE activation: no one can be filled
    assert all(len(a.fills) == 0 for a in ag.values())

    sim.on_trade(20, 100.0, 3, -1)       # seller hits 100 for 3
    # A: fills min(2, 3) = 2.  B: queue 5 -> 2, nothing reaches us.  C: no trade-through.
    assert [len(ag[k].fills) for k in "ABC"] == [1, 0, 0]
    assert ag["A"].pos == 2

    sim.on_message(30, [F], [T], [100.0], [2])     # level 100: 5 -> 2, all explained by trade
    sim.on_trade(40, 100.0, 3, -1)       # 2 ahead consumed, 1 reaches B
    assert ag["B"].fills[-1][3] == pytest.approx(1.0)
    assert ag["B"].pos == pytest.approx(1.0)
    assert len(ag["C"].fills) == 0

    sim.on_trade(50, 99.0, 0.5, -1)      # trade THROUGH 100: everyone at 100 is filled
    assert ag["B"].pos == pytest.approx(2.0)       # B's remaining 1 filled completely
    assert ag["C"].pos == pytest.approx(2.0)       # C's first and only kind of fill
    assert ag["C"].fills[-1][5] is True            # flagged as a through fill


def test_optimistic_fill_split_into_supported_and_phantom():
    sim, ag = make()
    snapshot(sim, 0)
    sim.on_trade(20, 100.0, 6, -1)       # 5 ahead, a 6-lot trade
    t, side, px, qty, mid, through, supported = ag["A"].fills[0]
    assert (qty, supported) == (2, 1)    # A takes 2: 1 queue-supported, 1 phantom
    assert ag["B"].fills[0][3] == 1      # B independently takes exactly the supported 1


def test_cancel_takes_effect_only_after_latency():
    sim, ag = make()
    snapshot(sim, 0)                     # bids live at t=10 at price 100
    sim.on_message(20, [F], [T], [100.5], [3])     # new best bid 100.5: cancel 100, live at 30
    sim.on_trade(25, 100.0, 1, -1)       # the old order is still live until t=30
    assert ag["A"].pos == 1
    sim.on_trade(35, 100.0, 1, -1)       # old order cancelled at 30; new bid at 100.5 live at 30
    # A sell at 100 is THROUGH the new 100.5 bid, so that order fills completely (2).
    # If the cancelled 100 order were still live it would also fill: prices prove it isn't.
    assert ag["A"].pos == 3
    assert [f[2] for f in ag["A"].fills] == [100.0, 100.5]


def test_post_only_rejection_when_market_moves_during_latency():
    sim, ag = make()
    snapshot(sim, 0)                     # bid decided at 100, live at 10
    # Before activation the asks drop to 100: our bid at 100 would cross.
    sim.on_message(5, [F, F], [F, F], [101.0, 100.0], [0, 3])
    sim.advance(10)
    assert all(a.n_reject >= 1 for a in ag.values())
    assert all(len(a.live[1]) == 0 or a.live[1][0].price != 100.0 for a in ag.values())


def test_inventory_limit_stops_the_bid():
    specs = [AgentSpec("A", "A", latency_us=10)]
    sim = Simulator(specs, q=2.0, qmax=2.0)        # limit = one order
    a = sim.agents[0]
    snapshot(sim, 0)
    sim.on_trade(20, 100.0, 5, -1)       # bid filled: position 2 = limit
    assert a.pos == 2
    assert a.intent[1] is None           # no new bid while at the limit
    assert a.intent[-1] is not None      # the ask keeps quoting


def test_rows_before_first_snapshot_are_ignored_and_reset_drops_orders():
    sim, ag = make()
    sim.on_message(-5, [F], [T], [90.0], [9])      # junk before the snapshot
    assert sim.book.best_bid() is None
    snapshot(sim, 0)
    assert sim.book.best_bid() == 100.0
    sim.on_message(50, [F], [T], [100.0], [6])     # ordinary update between snapshots
    snapshot(sim, 100)                   # a new snapshot after updates = reconnect / reset
    assert sim.stats["resets"] == 1
