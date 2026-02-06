import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: python read_npz.py <filename.npz>")
    sys.exit(1)

filename = sys.argv[1]

data = np.load(filename, allow_pickle=True)

with open(filename + '.txt', 'w') as f:
    f.write(f"NPZ File: {filename}\n")
    f.write(f"Keys: {list(data.files)}\n\n")
    
    for key in data.files:
        array = data[key]
        f.write(f"Key: {key}\n")
        f.write(f"Type: {type(array)}\n")
        f.write(f"Data type: {array.dtype}\n")
        f.write(f"Shape: {array.shape}\n")
        f.write(f"Content:\n{array}\n")
        f.write("-" * 50 + "\n\n")
