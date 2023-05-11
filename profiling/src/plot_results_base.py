import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

out = pd.read_csv("results/tims/ecoli.diadem.csv")

decoys = out[np.invert(out["decoy"])]
targets = out[out["decoy"]]

plt.hist(decoys["Score"], label="Decoys", color="red", alpha=0.2, bins=100)
plt.hist(targets["Score"], label="Targets", color="blue", alpha=0.2, bins=100)

plt.legend()
