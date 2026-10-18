"""
execution.py - the three fill rules, as pure functions.

Everything here concerns ONE hypothetical resting order and ONE historical event.
Keeping the rules pure (no hidden state) means each can be checked against hand-worked
examples in tests/test_execution.py.

Terms
  side          +1 for our bid, -1 for our ask
  rem           our order's unfilled quantity
  queue_ahead   displayed size estimated to be ahead of us at our price (L2 cannot
                observe this, so models B and the optimistic agent's shadow estimate it)
  through       the trade printed at a price WORSE than ours for the resting side
                (below our bid / above our ask). Price priority means every order at our
                price, including ours, must already have been consumed.
  at            the trade printed exactly at our price
"""


def classify(side, order_price, trade_price):
    """Return (through, at) for a trade that hits our side of the book."""
    if side == 1:
        return trade_price < order_price, trade_price == order_price
    return trade_price > order_price, trade_price == order_price


def fill_on_trade(model, side, order_price, rem, queue_ahead, trade_price, trade_qty):
    """
    Apply one trade (whose aggressor hits our side) to one resting order.

    Returns (fill_qty, new_queue_ahead, through, queue_supported_qty)

    queue_supported_qty is what the queue-aware rule would grant on this trade. For the
    optimistic model it lets us split its fill into supported and phantom volume on the
    SAME order (hypothesis H3).
    """
    through, at = classify(side, order_price, trade_price)
    if not (through or at):
        return 0.0, queue_ahead, False, 0.0

    if through:
        # The level was swept: all three models agree the order is filled completely.
        return rem, 0.0, True, rem

    # Trade exactly at our price: volume consumes the queue ahead first.
    used = min(queue_ahead, trade_qty)
    new_qa = queue_ahead - used
    supported = min(rem, trade_qty - used)

    if model == "A":
        fill = min(rem, trade_qty)          # ignores everyone ahead of us
    elif model == "B":
        fill = supported                    # only the volume left after the queue
    elif model == "C":
        fill = 0.0                          # needs a trade-through
    else:
        raise ValueError(f"unknown model {model!r}")
    return fill, new_qa, False, supported


def queue_after_update(rule, queue_ahead, old, new, traded):
    """
    Update queue_ahead when the displayed size at our price changes from old to new.

    `traded` is volume already attributed to trades at this price since the last update
    (those trades have already reduced queue_ahead in fill_on_trade). Whatever decrease
    is left over must be cancellations, and L2 data cannot say where in the queue they
    were. Each rule is one assumption about that:

      prop   cancellations are spread evenly through the queue (base case)
      back   all cancellations come from behind us (optimistic)
      front  all cancellations come from ahead of us (pessimistic)

    Increases are new orders, which join behind us: no change.
    Whatever the rule, the queue ahead cannot exceed the displayed size.
    """
    cancelled = max(0.0, old - new - traded)
    if cancelled > 0.0 and queue_ahead > 0.0:
        if rule == "prop":
            base = old - traded                       # size present before the cancels
            queue_ahead -= cancelled * queue_ahead / base
        elif rule == "front":
            queue_ahead -= cancelled
        elif rule != "back":
            raise ValueError(f"unknown rule {rule!r}")
    return max(0.0, min(queue_ahead, new))
