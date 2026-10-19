import sys

import gymnasium
import matplotlib
import numpy
import pandas
import torch
import traci


print("=" * 50)
print("Traffic Signal RL - Environment Check")
print("=" * 50)

print(f"Python       : {sys.version.split()[0]}")
print(f"NumPy        : {numpy.__version__}")
print(f"Pandas       : {pandas.__version__}")
print(f"Matplotlib   : {matplotlib.__version__}")
print(f"Gymnasium    : {gymnasium.__version__}")
print(f"PyTorch      : {torch.__version__}")
print(f"TraCI        : {traci.__version__}")

print()
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

print()
print("All Python packages imported successfully.")
print("Environment setup check completed.")
print("=" * 50)