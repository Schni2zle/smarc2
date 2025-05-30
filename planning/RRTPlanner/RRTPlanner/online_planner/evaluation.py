import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
baseline = pd.read_csv("metrics_baseline.csv")
safe = pd.read_csv("metrics_safe.csv")

# Calculate metrics
baseline["path_ratio"] = baseline["path_length_back"] / baseline["path_length"]
safe["path_ratio"] = safe["path_length"] / safe["path_length"]

baseline["total_compute"] = baseline["computation_time"] + baseline["back_computation_time"]
safe["total_compute"] = safe["computation_time"] + safe["back_computation_time"]

# Summary statistics
print("=== Baseline Planner ===")
print(baseline[["path_ratio", "total_compute"]].describe())

print("\n=== Safe Planner ===")
print(safe[["path_ratio", "total_compute"]].describe())

# Battery safety margin check (40% return budget)
def return_feasible(df, budget=0.4):
    return (df["path_length_back"] <= budget * (df["path_length"] + df["path_length_back"])).mean()

safe_return_success = return_feasible(safe)
baseline_return_success = return_feasible(baseline)

print(f"\nReturn Feasibility (within 40% budget):")
print(f"Safe Planner: {100 * safe_return_success:.2f}%")
print(f"Baseline Planner: {100 * baseline_return_success:.2f}%")

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

# 1. Path Ratios
axs[0].hist(safe["path_ratio"], bins=30, alpha=0.7, label='Safe')
axs[0].hist(baseline["path_ratio"], bins=30, alpha=0.7, label='Baseline')
axs[0].set_title("Return-to-Forward Path Length Ratio")
axs[0].set_xlabel("Ratio")
axs[0].set_ylabel("Frequency")
axs[0].legend()

# 2. Compute Time
axs[1].hist(safe["total_compute"], bins=30, alpha=0.7, label='Safe')
axs[1].hist(baseline["total_compute"], bins=30, alpha=0.7, label='Baseline')
axs[1].set_title("Total Compute Time")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Frequency")
axs[1].legend()

# 3. Boxplot of path ratios
axs[2].boxplot([baseline["path_ratio"], safe["path_ratio"]], labels=["Baseline", "Safe"])
axs[2].set_title("Path Ratio Comparison (Boxplot)")
axs[2].set_ylabel("backward / forward")

# 4. Boxplot of total compute time
axs[3].boxplot([baseline["total_compute"], safe["total_compute"]], labels=["Baseline", "Safe"])
axs[3].set_title("Total Compute Time (Boxplot)")
axs[3].set_ylabel("Time (s)")

plt.tight_layout()
plt.show()
