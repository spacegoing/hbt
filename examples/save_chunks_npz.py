import numpy as np
import os


def process_large_npz_in_chunks(npz_file, chunk_mem_size, save_dir,
                                base_filename):
  with np.load(npz_file, mmap_mode='r') as data:
    data = data['data']  # file is saved by np.savez(filename, data=data)
    num_rows, num_cols = data.shape
    row_mem_size = data.dtype.itemsize * num_cols  # Memory size of a single row
    chunksize = chunk_mem_size // row_mem_size  # Calculate number of rows per chunk based on memory size

    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    num_chunks = (num_rows + chunksize - 1) // chunksize

    for i in range(num_chunks):
      start = i * chunksize
      end = min((i + 1) * chunksize, num_rows)
      chunk = data[start:end, :]
      np.savez(os.path.join(save_dir, f"{base_filename}_{i}.npz"), data=chunk)


# Example usage
npz_file = "btcusdt_20230405.npz"  # Replace with your actual NPZ file path
chunk_mem_size = 2 * 1024**2  # 2GB in bytes
save_dir = "test_chunks"
base_filename = "btcusdt_20230405"
process_large_npz_in_chunks(npz_file, chunk_mem_size, save_dir, base_filename)

#*

# Load the original data
original_data = np.load(npz_file)['data']

# Load all chunked npz files
chunk_files = sorted([
    './test_chunks/' + f
    for f in os.listdir('./test_chunks')
    if f.endswith('.npz')
])

# Combine the chunks into one ndarray
combined_data = np.vstack(
    [np.load(chunk_file)['data'] for chunk_file in chunk_files])

# Compare the combined data with the original data
are_identical = np.array_equal(original_data, combined_data)

print("Are the original data and the combined data identical?", are_identical)
