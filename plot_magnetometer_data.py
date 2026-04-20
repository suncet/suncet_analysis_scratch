import pandas as pd
import matplotlib.pyplot as plt
import os

# path
csv_file = os.path.join(os.getenv("suncet_data"), "test_data", "2026-04-06_post_vibe_cpt", "suncet_tr_20260406_125316.csv")

# read
df = pd.read_csv(csv_file)

# parse datetime (your format)
df["datetime"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")

# clean numeric
for col in ["X", "Y", "Z"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["datetime", "X", "Y", "Z"])

# ---- stacked subplots ----
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(df["datetime"], df["X"], color="tomato")
axes[0].set_ylabel("X (µT)")
axes[0].set_title("Magnetometer Data")

axes[1].plot(df["datetime"], df["Y"], color="limegreen")
axes[1].set_ylabel("Y (µT)")

axes[2].plot(df["datetime"], df["Z"], color="dodgerblue")
axes[2].set_ylabel("Z (µT)")
axes[2].set_xlabel("Time")

plt.tight_layout()
plt.show()