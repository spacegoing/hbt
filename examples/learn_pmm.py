import numpy as np
import random
import pandas as pd
from numba import njit
from hftbacktest import HftBacktest, FeedLatency, SquareProbQueueModel, Linear, Stat, BUY, SELL, GTX

seed = 23
# Set seed for NumPy
np.random.seed(seed)
# Set seed for Pandas
random.seed(seed)


#* Strategy
def simple_two_sided_quote(hbt, stat):
  max_position = 5
  half_spread = hbt.tick_size * 20
  skew = 1
  order_qty = 0.1
  last_order_id = -1
  order_id = 0

  while hbt.run:
    # Check every 0.1s
    if not hbt.elapse(1e5):
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


# Backtest
# This backtest assumes market maker rebates.
# https://www.binance.com/kz/support/announcement/binance-upgrades-usd%E2%93%A2-margined-futures-liquidity-provider-program-2023-04-04-01007356e6514df3811b0c80ab8c83bf

hbt = HftBacktest(
    [
        'test_chunks/btcusdt_20230405_0.npz',
        'test_chunks/btcusdt_20230405_1.npz',
        'test_chunks/btcusdt_20230405_2.npz',
        'test_chunks/btcusdt_20230405_3.npz',
        'test_chunks/btcusdt_20230405_4.npz',
        'test_chunks/btcusdt_20230405_5.npz',
        'test_chunks/btcusdt_20230405_6.npz',
    ],
    # hbt = HftBacktest([
    #     'btcusdt_20230405.npz',
    # ],
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

# Viz
stat.summary(capital=2000)

if __name__ == "__main__":
  #* Debug
  a = np.load('chunked_balance.npy')
  unequal_indices = np.argwhere(a != stat.balance)
  first_unequal_index = unequal_indices[0, 0]
  print(first_unequal_index)

  pos = np.load('o_position.npy')
  unequal_indices = np.argwhere(pos != stat.position)
  first_unequal_index = unequal_indices[0, 0]
  print(first_unequal_index)

  # dt = stat.datetime()
  # dt[1478]
  # Out[14]: 1680653149332808
  # Out[18]: Timestamp('2023-04-05 00:05:49.332808+0000', tz='UTC')

  # For chunked timestamp
  # In [11]: stat.timestamp[1478]
  # Out[11]: 1680653149379907

  row = 0
  for i in range(7):
    filename = f'test_chunks/btcusdt_20230405_{i}.npz'
    data = np.load(filename)['data']
    print(filename)
    date = np.where(data[:, 1] > 0, data[:, 1], data[:, 2])
    unequal_indices = np.argwhere(date.astype(int) > 1680653112209163)
    if np.size(unequal_indices) == 0:
      print('Not found !!!')
      row += data.shape[0]
      # 218450 + 263
    else:
      first_unequal_index = unequal_indices[0, 0]
      print('FFFFFFFFFFFFFFFFF')
      print(first_unequal_index)

  cked_5_diff = data[263 - 5:263 + 5, :]
  np.save('slice_5.npy', cked_5_diff)

  d = np.load('btcusdt_20230405.npz')['data']
  idx = 218450 + 263
  td = d[idx - 5:idx + 5, :]

  idx = 17400
  date = np.where(data[idx, 1] > 0, data[idx, 1], data[idx, 2])
  t = pd.to_datetime(date, utc=True, unit='us')
  print(t)

#* debug timestamp
ots = np.load('o_ts.npy')
# np.save('o_ts.npy', ots)
cts = np.array(stat.timestamp)
# np.save('c_ts.npy', cts)
unequal_indices = np.argwhere(cts != ots)

idx = unequal_indices[0, 0]
print(idx)
# 1324
print(ots[idx], cts[idx])
# 1680653112209163 1680653112213197
# 2023-04-05 00:05:12.209163+00:00
# 2023-04-05 00:05:12.213197+00:00
# print(pd.to_datetime(1680653112209163 , utc=True, unit='us'))
print(ots[idx - 5:idx + 5])
print(cts[idx - 5:idx + 5])

