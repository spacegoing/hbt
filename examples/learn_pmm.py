from numba import njit
from hftbacktest import HftBacktest, FeedLatency, SquareProbQueueModel, Linear, Stat, BUY, SELL, GTX


#* Strategy
@njit
def simple_two_sided_quote(hbt, stat):
  max_position = 5
  half_spread = hbt.tick_size * 20
  skew = 1
  order_qty = 0.1
  last_order_id = -1
  order_id = 0

  while hbt.run:
    # Check every 0.1s
    if not hbt.elapse(0.1 * 1e6):
      return False

    # Clear cancelled, filled or expired orders.
    hbt.clear_inactive_orders()

    # Obtain the current mid-price and compute the reservation price.
    mid_price = (hbt.best_bid + hbt.best_ask) / 2.0
    reservation_price = mid_price - skew * hbt.position * hbt.tick_size

    buy_order_price = reservation_price - half_spread
    sell_order_price = reservation_price + half_spread

    last_order_id = -1
    # Cancel all outstanding orders
    for order in hbt.orders.values():
      if order.cancellable:
        hbt.cancel(order.order_id)
        last_order_id = order.order_id

    # All order requests are considered to be requested at the same time.
    # Wait until one of the order cancellation responses is received.
    if last_order_id >= 0:
      hbt.wait_order_response(last_order_id)

    # Clear cancelled, filled or expired orders.
    hbt.clear_inactive_orders()

    if hbt.position < max_position:
      # Submit a new post-only limit bid order.
      order_id += 1
      hbt.submit_buy_order(order_id, buy_order_price, order_qty, GTX)
      last_order_id = order_id

    if hbt.position > -max_position:
      # Submit a new post-only limit ask order.
      order_id += 1
      hbt.submit_sell_order(order_id, sell_order_price, order_qty, GTX)
      last_order_id = order_id

    # All order requests are considered to be requested at the same time.
    # Wait until one of the order responses is received.
    if last_order_id >= 0:
      hbt.wait_order_response(last_order_id)

    # Record the current state for stat calculation.
    stat.record(hbt)
  return True


#* Backtest
# This backtest assumes market maker rebates.
# https://www.binance.com/kz/support/announcement/binance-upgrades-usd%E2%93%A2-margined-futures-liquidity-provider-program-2023-04-04-01007356e6514df3811b0c80ab8c83bf

hbt = HftBacktest([
    'btcusdt_20230405.npz',
],
                  tick_size=0.01,
                  lot_size=0.001,
                  maker_fee=-0.00005,
                  taker_fee=0.0007,
                  order_latency=FeedLatency(),
                  queue_model=SquareProbQueueModel(),
                  asset_type=Linear,
                  snapshot='btcusdt_20230404_eod.npz')

stat = Stat(hbt)
simple_two_sided_quote(hbt, stat.recorder)

#* Viz
stat.summary(capital=2000)
