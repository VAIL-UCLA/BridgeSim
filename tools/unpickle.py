import pickle
import pprint 
import sys

if len(sys.argv) != 2:
    print("Usage: python unpickle.py <filename.pkl>")
    sys.exit(1)

filename = sys.argv[1]

with open(filename, 'rb') as f:
    data = pickle.load(f)

with open(filename + '.txt', 'w') as f:
    pprint.pprint(data, stream=f)
