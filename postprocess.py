#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# 1) Load
df = pd.read_csv("results.csv")

# 2) Pivot GFLOP/s vs n for each k
pivot = df.pivot(index="n", columns="k", values="gflops")
pivot.plot(marker="o")
plt.title("HODLR-MVM GFLOP/s vs matrix size")
plt.xlabel("n")
plt.ylabel("GFLOP/s")
plt.xscale("log", base=2)
plt.grid(True, which="both", ls="--")
plt.legend(title="k")
plt.tight_layout()
plt.show()

# 3) Save processed summary if desired
df.to_excel("results.xlsx", index=False)