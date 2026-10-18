"""
book.py - level-2 order book reconstruction.

A level-2 (L2) book stores one number per price: the total displayed size resting there.
It cannot tell you how many orders make up that size or in what order they arrived.
That limitation is the reason execution models are needed at all.

Tardis rows give the NEW absolute size at a price (not a change). Size 0 removes the
level. Snapshot rows replace the whole book.

Finding the best price: a dict gives O(1) updates but max() over its keys is O(n).
We keep a heap of prices alongside each dict and delete lazily: stale heap entries
(prices no longer in the dict) are discarded only when they reach the top. Best-price
queries are then O(log n) amortised. The heap is rebuilt if stale entries pile up.
"""
import heapq


class OrderBook:
    def __init__(self):
        self.bids = {}      # price -> displayed size
        self.asks = {}
        self._bh = []       # max-heap of bid prices (stored negated)
        self._ah = []       # min-heap of ask prices

    def reset(self):
        self.bids.clear()
        self.asks.clear()
        self._bh.clear()
        self._ah.clear()

    def level(self, is_bid, price):
        return (self.bids if is_bid else self.asks).get(price, 0.0)

    def apply(self, is_bid, price, amount):
        """Set the size at `price`. Returns the previous size (0.0 if absent)."""
        side = self.bids if is_bid else self.asks
        old = side.get(price, 0.0)
        if amount == 0.0:
            if price in side:
                del side[price]
        else:
            new_level = price not in side
            side[price] = amount               # before any rebuild, so the rebuild includes it
            if new_level:
                heap = self._bh if is_bid else self._ah
                heapq.heappush(heap, -price if is_bid else price)
                if len(heap) > 4 * len(side) + 1024:      # too many stale entries
                    heap[:] = [-p for p in side] if is_bid else list(side)
                    heapq.heapify(heap)
        return old

    def best_bid(self):
        h, b = self._bh, self.bids
        while h and -h[0] not in b:
            heapq.heappop(h)
        return -h[0] if h else None

    def best_ask(self):
        h, a = self._ah, self.asks
        while h and h[0] not in a:
            heapq.heappop(h)
        return h[0] if h else None
