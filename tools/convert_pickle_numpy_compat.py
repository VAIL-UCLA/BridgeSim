#!/usr/bin/env python
"""Convert numpy 2.x pickles to numpy 1.x compatible format"""

import pickle
import sys
from pathlib import Path

def convert_pickle(input_path, output_path):
    """Load pickle with numpy 2.x and save with protocol compatible with numpy 1.x"""
    print(f"Loading {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)  # Protocol 4 is compatible

    print("Done!")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_pickle_numpy_compat.py input.pkl output.pkl")
        sys.exit(1)

    convert_pickle(sys.argv[1], sys.argv[2])
