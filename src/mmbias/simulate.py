"""
simulate.py - replay one day of historical data with simulated market makers.

Design
------
* One replay of the historical data drives many agents at once. Each agent is the same
  strategy under a different execution model or latency. Agents never see each other:
  each lives in its own counterfactual world on the same historical market.
* Event order: exchange timestamp. On equal timestamps the book update comes FIRST and
  the trade after it: on real Binance USD-M data a trade stamped with the same
  millisecond as a book message is reflected in the NEXT message, not that one
  (src/mmbias/validate.py, research_plan.md amendment 5). Traded volume is therefore
  netted against the depletion of the message that actually shows it. The pre-registered
  opposite rule is kept as an option (trades_first_on_ties) for a sensitivity run.
* Accounting order: every event gets a sequence number, and fills and mid changes are
  ordered by it rather than by timestamp, so the PnL decomposition is exact whatever the
  tie rule.
* Latency: decisions schedule actions at t + L on a global heap. Before any event at
  time t, every action due at or before t is applied. An order is fillable from its
  activation until its cancellation takes effect.
* No look-ahead: a decision at time t uses only the book and fills up to t. Nothing in
  the loop reads ahead. tests/test_lookahead.py checks this by truncating the data.
* Our orders are hypothetical: they never change the historical book or trades
  (the no-market-impact assumption, deliberately small size).
* Memory: book and trade files are streamed (events.py). What grows with the day is the
  record of fills and mid changes, which analysis needs; it is kept in typed arrays
  (FillLog), about 50 bytes per fill rather than about 250 for a tuple of Python objects.
"""
import heapq
from array import array
from dataclasses import dataclass

import numpy as np

from .book import OrderBook
from .events import iter_book_messages, iter_trades
from .execution import fill_on_trade, queue_after_update

EPS = 1e-12
PENDING, LIVE, DONE = 0, 1, 2


class FillLog:
    """Fills stored column-wise. Indexing and iteration give
    (t, side, price, qty, mid, through, supported) tuples, as a list of tuples would."""
    __slots__ = ("t", "side", "price", "qty", "mid", "through", "supported", "seq")

    def __init__(self):
        self.t, self.side, self.through = array("q"), array("b"), array("b")
        self.price, self.qty, self.mid, self.supported = array("d"), array("d"), array("d"), array("d")
        self.seq = array("q")                   # event sequence number (accounting order)

    def append(self, t, side, price, qty, mid, through, supported, seq=0):
        self.seq.append(seq)
        self.t.append(t)
        self.side.append(side)
        self.price.append(price)
        self.qty.append(qty)
        self.mid.append(float("nan") if mid is None else mid)
        self.through.append(1 if through else 0)
        self.supported.append(supported)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        return (self.t[i], self.side[i], self.price[i], self.qty[i], self.mid[i],
                bool(self.through[i]), self.supported[i])

    def __iter__(self):
        return (self[i] for i in range(len(self.t)))

    def __eq__(self, other):
        return list(self) == list(other)

    def to_arrays(self):
        return dict(t=np.array(self.t, dtype=np.int64), side=np.array(self.side, dtype=float),
                    price=np.array(self.price), qty=np.array(self.qty), mid=np.array(self.mid),
                    through=np.array(self.through, dtype=bool), supported=np.array(self.supported),
                    seq=np.array(self.seq, dtype=np.int64))


class Order:
    __slots__ = ("agent", "side", "price", "qty", "rem", "t_live", "t_cancel", "qa", "state")

    def __init__(self, agent, side, price, qty, t_live):
        self.agent, self.side, self.price = agent, side, price
        self.qty = self.rem = qty
        self.t_live, self.t_cancel = t_live, None
        self.qa = 0.0
        self.state = PENDING


@dataclass
class AgentSpec:
    name: str
    model: str            # "A" optimistic, "B" queue-aware, "C" conservative
    qrule: str = "prop"   # cancellation rule for B ("prop", "back", "front")
    latency_us: int = 50_000


class Agent:
    def __init__(self, spec, q, qmax, record_decisions=False):
        self.name, self.model, self.qrule = spec.name, spec.model, spec.qrule
        self.lat = spec.latency_us
        self.q, self.qmax = q, qmax
        self.pos = 0.0
        self.intent = {1: None, -1: None}   # the order each side is meant to have
        self.live = {1: [], -1: []}
        self.fills = FillLog()  # (t, side, price, qty, mid, through, supported)
        self.n_place = self.n_cancel = self.n_reject = 0
        self.n_live = self.n_full = self.n_partial = 0   # orders live / fully filled / cancelled part-filled
        self.max_abs_pos = 0.0
        self.qty_at_activation = 0.0    # filled in the same millisecond the order went live
        self.last = None
        self.dirty = False
        # (t, side, price) of every placement, kept only for the look-ahead test
        self.decision_log = [] if record_decisions else None

    def record_fill(self, t, side, price, qty, mid, through, supported, seq=0):
        self.fills.append(t, side, price, qty, mid, through, supported, seq)
        self.pos += side * qty
        self.max_abs_pos = max(self.max_abs_pos, abs(self.pos))
        self.dirty = True
        self.last = None

    def decide(self, t, sim):
        """Join-the-touch quoting with an inventory limit. Uses only state at time t."""
        self.dirty = False
        bb, ba = sim.bb, sim.ba
        state = (bb, ba, self.pos)
        if state == self.last:
            return
        self.last = state
        crossed = bb is not None and ba is not None and bb >= ba
        for side in (1, -1):
            if crossed:
                desired = None
            elif side == 1:
                desired = bb if (bb is not None and self.pos + self.q <= self.qmax + EPS) else None
            else:
                desired = ba if (ba is not None and self.pos - self.q >= -self.qmax - EPS) else None
            cur = self.intent[side]
            if cur is not None and cur.state == DONE:
                cur = self.intent[side] = None
            if desired == (cur.price if cur is not None else None):
                continue                      # keep the order and its queue position
            if cur is not None:
                cur.t_cancel = t + self.lat
                sim.schedule(cur.t_cancel, 1, cur)
                self.intent[side] = None
            if desired is not None:
                o = Order(self, side, desired, self.q, t + self.lat)
                sim.schedule(o.t_live, 0, o)
                self.intent[side] = o
                self.n_place += 1
                if self.decision_log is not None:
                    self.decision_log.append((t, side, desired))


class Simulator:
    def __init__(self, specs, q, qmax, record_decisions=False):
        self.agents = [Agent(s, q, qmax, record_decisions) for s in specs]
        self.book = OrderBook()
        self.sched, self.seq = [], 0
        self.registry = {}     # (is_bid, price) -> orders whose queue position we track
        self.acc = {}          # (is_bid, price) -> traded volume since that level's last update
        self.bb = self.ba = self.mid = None
        self.mid_t, self.mid_v, self.mid_seq = array("q"), array("d"), array("q")
        self.n_ev = 0          # event sequence number: orders fills against mid changes
        self.started = False
        self.prev_snap = False
        self.tick_tol = 1e-9
        self.stats = dict(messages=0, rows=0, resets=0, crossed=0, locked=0, book_states=0,
                          missing_delete=0, trades=0, trades_inside=0, trades_no_book=0,
                          first_ts=None, last_ts=None)

    # ---- scheduling -------------------------------------------------------------------
    def schedule(self, t, kind, order):
        heapq.heappush(self.sched, (t, self.seq, kind, order))
        self.seq += 1

    def advance(self, t):
        s = self.sched
        while s and s[0][0] <= t:
            tt, _, kind, o = heapq.heappop(s)
            if kind == 0:
                self._activate(o, tt)
            else:
                self._cancel(o)

    def _register(self, o):
        self.registry.setdefault((o.side == 1, o.price), []).append(o)

    def _unregister(self, o):
        key = (o.side == 1, o.price)
        lst = self.registry.get(key)
        if lst is not None:
            try:
                lst.remove(o)
            except ValueError:
                pass
            if not lst:
                del self.registry[key]
                self.acc.pop(key, None)

    def _activate(self, o, t):
        if o.state != PENDING:
            return
        ag = o.agent
        if (o.side == 1 and self.ba is not None and o.price >= self.ba) or \
           (o.side == -1 and self.bb is not None and o.price <= self.bb):
            o.state = DONE                               # post-only: would have crossed
            ag.n_reject += 1
            if ag.intent[o.side] is o:
                ag.intent[o.side] = None
            ag.last, ag.dirty = None, True
            return
        o.state = LIVE
        ag.n_live += 1
        o.qa = self.book.level(o.side == 1, o.price)     # join the back of the queue
        ag.live[o.side].append(o)
        if ag.model in ("A", "B"):
            self._register(o)

    def _cancel(self, o):
        if o.state == LIVE:
            o.agent.live[o.side].remove(o)
            o.agent.n_cancel += 1
            if o.rem < o.qty - EPS:
                o.agent.n_partial += 1
            self._unregister(o)
        o.state = DONE

    def _drop_all_orders(self):
        """A book reset means a data gap: assume no fills, start quoting afresh."""
        for ag in self.agents:
            for side in (1, -1):
                for o in ag.live[side]:
                    o.state = DONE
                ag.live[side].clear()
                if ag.intent[side] is not None:
                    ag.intent[side].state = DONE
                    ag.intent[side] = None
            ag.last, ag.dirty = None, True
        self.registry.clear()
        self.acc.clear()
        self.sched.clear()

    # ---- events -----------------------------------------------------------------------
    def on_trade(self, t, price, qty, aggr):
        self.advance(t)
        st = self.stats
        st["trades"] += 1
        self.n_ev += 1
        if aggr == 0:
            return
        if self.bb is None or self.ba is None:
            st["trades_no_book"] += 1
        elif (aggr == 1 and price < self.ba - self.tick_tol) or \
             (aggr == -1 and price > self.bb + self.tick_tol):
            st["trades_inside"] += 1          # buy below best ask / sell above best bid
        hit = -aggr                           # a seller-initiated trade hits bids (+1)
        key = (hit == 1, price)
        if key in self.registry:
            self.acc[key] = self.acc.get(key, 0.0) + qty
        changed = False
        for ag in self.agents:
            lst = ag.live[hit]
            if not lst:
                continue
            for o in list(lst):
                if (hit == 1 and price > o.price) or (hit == -1 and price < o.price):
                    continue
                f, o.qa, through, supported = fill_on_trade(
                    ag.model, hit, o.price, o.rem, o.qa, price, qty)
                if f > EPS:
                    ag.record_fill(t, hit, o.price, f, self.mid, through, supported, self.n_ev)
                    if t == o.t_live:
                        ag.qty_at_activation += f
                    o.rem -= f
                    changed = True
                    if o.rem <= EPS:
                        ag.n_full += 1
                        o.state = DONE
                        lst.remove(o)
                        self._unregister(o)
                        if ag.intent[hit] is o:
                            ag.intent[hit] = None
        if changed:
            for ag in self.agents:
                if ag.dirty:
                    ag.decide(t, self)

    def on_message(self, ts, snap, bid, px, amt):
        self.advance(ts)
        st = self.stats
        st["messages"] += 1
        self.n_ev += 1
        book, reg, acc = self.book, self.registry, self.acc
        for s, b, p, a in zip(snap, bid, px, amt):
            if s and not self.prev_snap:
                if self.started:
                    st["resets"] += 1
                    self._drop_all_orders()
                self.started = True
                book.reset()
            self.prev_snap = s
            if not self.started:
                continue                       # rows before the first snapshot
            st["rows"] += 1
            old = book.apply(b, p, a)
            if a == 0.0 and old == 0.0:
                st["missing_delete"] += 1
            key = (b, p)
            if key in reg:
                traded = acc.pop(key, 0.0)
                for o in reg[key]:
                    rule = "prop" if o.agent.model == "A" else o.agent.qrule
                    o.qa = queue_after_update(rule, o.qa, old, a, traded)
        if not self.started:
            return
        if st["first_ts"] is None:
            st["first_ts"] = ts
        st["last_ts"] = ts
        bb, ba = book.best_bid(), book.best_ask()
        changed = (bb != self.bb) or (ba != self.ba)
        self.bb, self.ba = bb, ba
        if bb is not None and ba is not None:
            st["book_states"] += 1
            if bb > ba:
                st["crossed"] += 1
            elif bb == ba:
                st["locked"] += 1
            mid = 0.5 * (bb + ba)
            if mid != self.mid:
                self.mid = mid
                self.mid_t.append(ts)
                self.mid_v.append(mid)
                self.mid_seq.append(self.n_ev)
        if changed:
            for ag in self.agents:
                ag.decide(ts, self)
        else:
            for ag in self.agents:
                if ag.dirty:
                    ag.decide(ts, self)


def run_day(book_path, trades_path, specs, q, qmax, chunksize=250_000, until_ts=None,
            record_decisions=False, trades_first_on_ties=False):
    """Replay one day. `until_ts` truncates the data (used by the look-ahead test).

    Both files are streamed and merged on exchange timestamp, so memory does not grow
    with the number of trades or book rows in the day. On equal timestamps the book
    message goes first, unless trades_first_on_ties (the pre-registered rule, kept for
    the sensitivity run).
    """
    sim = Simulator(specs, q, qmax, record_decisions)
    sim.stats["tie_rule"] = "trades_first" if trades_first_on_ties else "book_first"
    trades = iter_trades(trades_path, chunksize, sim.stats)
    nxt = next(trades, None)
    for ts, lts, snap, bid, px, amt in iter_book_messages(book_path, chunksize):
        if until_ts is not None and ts > until_ts:
            break
        while nxt is not None and (nxt[0] < ts or (trades_first_on_ties and nxt[0] == ts)):
            sim.on_trade(*nxt)
            nxt = next(trades, None)
        sim.on_message(ts, snap, bid, px, amt)
    while nxt is not None and (until_ts is None or nxt[0] <= until_ts):
        sim.on_trade(*nxt)
        nxt = next(trades, None)
    return sim
