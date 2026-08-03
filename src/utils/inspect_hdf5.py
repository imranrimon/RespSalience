import h5py
import sys

def inspect_h5(path):
    try:
        with h5py.File(path, 'r') as f:
            print(f"Keys in {path}: {list(f.keys())}")
            # Recursively print
            def print_attrs(name, obj):
                print(name, obj)
            f.visititems(print_attrs)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_h5(sys.argv[1])
