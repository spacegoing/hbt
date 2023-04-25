import gzip
import numpy as np
from hftbacktest.data.utils import binancefutures
import pandas as pd

with gzip.open('usdm/btcusdt_20230404.dat.gz', 'r') as f:
  for i in range(20):
    line = f.readline()
    print(line)

#* Data Preparation
data = binancefutures.convert('usdm/btcusdt_20230404.dat.gz')
np.savez('btcusdt_20230404', data=data)
binancefutures.convert(
    'usdm/btcusdt_20230405.dat.gz', output_filename='btcusdt_20230405')

#* Viz Data
df = pd.DataFrame(
    data,
    columns=[
        'event', 'exch_timestamp', 'local_timestamp', 'side', 'price', 'qty'
    ])
df['event'] = df['event'].astype(int)
df['exch_timestamp'] = df['exch_timestamp'].astype(int)
df['local_timestamp'] = df['local_timestamp'].astype(int)
df['side'] = df['side'].astype(int)
print(df)

#* Eod Market Depth Snapshot
from hftbacktest.data.utils import create_last_snapshot

# Build 20230404 End of Day snapshot. It will be used for the initial snapshot for 20230405.
data = create_last_snapshot(
    'btcusdt_20230404.npz', tick_size=0.01, lot_size=0.001)
np.savez('btcusdt_20230404_eod.npz', data=data)

# Build 20230405 End of Day snapshot.
# Due to the file size limitation, btcusdt_20230405.npz does not contain data for the entire day.
create_last_snapshot(
    'btcusdt_20230405.npz',
    tick_size=0.01,
    lot_size=0.001,
    initial_snapshot='btcusdt_20230404_eod.npz',
    output_snapshot_filename='btcusdt_20230405_eod')

#* Viz Eod
df = pd.DataFrame(data, columns=['event', 'exch_timestamp', 'local_timestamp', 'side', 'price', 'qty'])
df['event'] = df['event'].astype(int)
df['exch_timestamp'] = df['exch_timestamp'].astype(int)
df['local_timestamp'] = df['local_timestamp'].astype(int)
df['side'] = df['side'].astype(int)
print(df)
