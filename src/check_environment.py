import sys

import matplotlib
import numpy
from PIL import Image


print("=" * 50)
print("Environment Check")
print("=" * 50)

print(f"Python: {sys.version.split()[0]}")
print(f"NumPy: {numpy.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Pillow: {Image.__version__}")

print("Environment setup completed successfully.")