"""Hand-calculated tests for the order book and the three fill rules."""
import random

import pytest

from mmbias.book import OrderBook
from mmbias.execution import classify, fill_on_trade, queue_after_update


# ---------------------------------------------------------------- order book
def test_book_best_prices_and_deletion():
    b = OrderBook()
    for p, a in [(100.0, 5), (99.0, 3), (98.0, 1)]:
        b.apply(True, p, a)
    for p, a in [(101.0, 2), (102.0, 4)]:
        b.apply(False, p, a)
    assert b.best_bid() == 100.0 and b.best_ask() == 101.0
    assert b.apply(True, 100.0, 0.0) == 5          # returns the old size
    assert b.best_bid() == 99.0                     # best recomputed after deletion
    b.apply(True, 100.5, 7)                         # new level inside the spread
    assert b.best_bid() == 100.5
    assert b.level(True, 100.5) == 7 and b.level(True, 50.0) == 0.0


def test_book_reset():
    b = OrderBook()
    b.apply(True, 100.0, 1)
    b.reset()
    assert b.best_bid() is None and b.best_ask() is None


def test_book_matches_brute_force_under_random_updates():
    """10,000 random updates on a narrow price range: best price always equals max()."""
    rnd = random.Random(1)
    b, ref = OrderBook(), {}
    for _ in range(10_000):
        p = float(rnd.randint(900, 1000))
        a = 0.0 if rnd.random() < 0.45 else float(rnd.randint(1, 9))
        b.apply(True, p, a)
        if a == 0.0:
            ref.pop(p, None)
        else:
            ref[p] = a
        assert b.best_bid() == (max(ref) if ref else None)


@pytest.mark.parametrize("is_bid", [True, False])
def test_heap_rebuild_keeps_the_level_being_added(is_bid):
    """Regression test for a bug found on real data (AVAXUSDT 2025-09-01).

    The heap is rebuilt from the dict when stale entries exceed 4 x levels + 1024. The
    rebuild used to run before the new level was written to the dict, so the level being
    added vanished from the heap and best_bid() skipped it (Tardis's own top-of-book
    disagreed with ours on 23 rows that day). Here every step adds a new best level and
    deletes the old one, so stale entries pile up under a live top and the rebuild is
    forced to happen on the insertion of a new best level.
    """
    sgn = 1 if is_bid else -1
    b = OrderBook()
    heap = b._bh if is_bid else b._ah
    get = b.best_bid if is_bid else b.best_ask
    best = 100.0
    b.apply(is_bid, best, 1.0)
    rebuilds = 0
    for k in range(1, 3000):
        new = round(100.0 + sgn * k / 100, 2)
        n_before = len(heap)
        b.apply(is_bid, new, 1.0)              # a new, better level is pushed
        rebuilds += len(heap) <= n_before      # a push that did not grow the heap = rebuild
        assert get() == new
        b.apply(is_bid, best, 0.0)             # the old best becomes a stale heap entry
        best = new
    assert rebuilds >= 2


def test_book_matches_brute_force_with_rebuilds():
    """30,000 random adds/deletes over many distinct prices (heap rebuilds occur); after
    each update the best price is checked, and at the end every level is drained in order."""
    rnd = random.Random(7)
    b, ref = OrderBook(), {}
    for _ in range(30_000):
        if rnd.random() < 0.5 and ref:
            p, a = rnd.choice(list(ref)), 0.0
        else:
            p, a = rnd.randint(0, 50_000) / 100, float(rnd.randint(1, 9))
        b.apply(False, p, a)
        if a == 0.0:
            ref.pop(p, None)
        else:
            ref[p] = a
        assert b.best_ask() == (min(ref) if ref else None)
    while ref:
        top = b.best_ask()
        assert top == min(ref) and b.level(False, top) == ref.pop(top)
        b.apply(False, top, 0.0)
    assert b.best_ask() is None


# ---------------------------------------------------------------- fill rules
def test_classify():
    assert classify(1, 100.0, 99.0) == (True, False)     # sold below our bid: through
    assert classify(1, 100.0, 100.0) == (False, True)
    assert classify(-1, 101.0, 102.0) == (True, False)   # bought above our ask: through
    assert classify(-1, 101.0, 100.0) == (False, False)


def test_trade_at_price_queue_not_cleared():
    # Our bid: price 100, 2 remaining, 5 displayed ahead. A seller trades 3 at 100.
    # A ignores the queue: fills 2. B: the 3 consume queue ahead (5 -> 2), fills 0.
    # C needs a trade-through: fills 0. The queue-supported volume is 0 for all.
    assert fill_on_trade("A", 1, 100.0, 2, 5, 100.0, 3) == (2, 2, False, 0)
    assert fill_on_trade("B", 1, 100.0, 2, 5, 100.0, 3) == (0, 2, False, 0)
    assert fill_on_trade("C", 1, 100.0, 2, 5, 100.0, 3) == (0, 2, False, 0)


def test_trade_at_price_queue_cleared_partial_fill():
    # 5 ahead, a 6-lot trade: 5 clears the queue, 1 reaches us. B fills 1 of our 2.
    # A fills 2, of which 1 is queue-supported and 1 is phantom.
    assert fill_on_trade("B", 1, 100.0, 2, 5, 100.0, 6) == (1, 0, False, 1)
    assert fill_on_trade("A", 1, 100.0, 2, 5, 100.0, 6) == (2, 0, False, 1)


def test_trade_through_fills_everyone_completely():
    for m in "ABC":
        assert fill_on_trade(m, 1, 100.0, 2, 5, 99.0, 0.1) == (2, 0.0, True, 2)
        assert fill_on_trade(m, -1, 101.0, 2, 5, 102.0, 0.1) == (2, 0.0, True, 2)


def test_trade_away_from_our_price_does_nothing():
    assert fill_on_trade("A", 1, 100.0, 2, 5, 100.5, 3) == (0.0, 5, False, 0.0)


# ---------------------------------------------------------------- queue updates
def test_queue_update_rules_hand_example():
    # 6 ahead of us. Level falls from 10 to 5; 3 of that was trades already applied.
    # Cancellations = 10 - 5 - 3 = 2. Size before cancels = 10 - 3 = 7.
    # prop:  6 - 2 * 6/7 = 4.2857   back: 6, capped at the new size 5   front: 6 - 2 = 4
    assert queue_after_update("prop", 6, 10, 5, 3) == pytest.approx(6 - 2 * 6 / 7)
    assert queue_after_update("back", 6, 10, 5, 3) == 5
    assert queue_after_update("front", 6, 10, 5, 3) == 4


def test_queue_update_increase_and_vanish():
    assert queue_after_update("prop", 3, 5, 8, 0) == 3     # new orders join behind us
    assert queue_after_update("back", 3, 5, 0, 0) == 0     # level emptied: we are first
    assert queue_after_update("front", 1, 5, 2, 0) == 0    # cannot go negative
