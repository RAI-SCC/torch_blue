import matplotlib.pyplot as plt
import numpy as np

# Example processor counts
processors = np.array([1, 2, 4, 8, 16])

# Example runtimes (replace with your actual data)
# For 512 samples
times_strategy1 = np.array([184.50, 94.04, 48.46, 26.10, 15.63])
times_strategy2 = np.array([184.50, 92.55, 46.40, 23.19, 11.65])
times_strategy3 = np.array([184.50, 94.04, 48.46, 24.13, 12.37])

# Compute speedups
speedup1 = times_strategy1[0] / times_strategy1
speedup2 = times_strategy2[0] / times_strategy2
speedup3 = times_strategy3[0] / times_strategy3

# Ideal linear speedup
ideal_speedup = processors  

# Plot
plt.figure(figsize=(7, 5))
plt.plot(processors, speedup1, marker='o', label="Sample Parallel")
plt.plot(processors, speedup2, marker='s', label="Data Parallel")
plt.plot(processors, speedup3, marker='^', label="Hybrid")
plt.plot(processors, ideal_speedup, 'k--', label="Ideal speedup")

# Formatting
plt.xscale("log", base=2)   # log scale for processors (optional)
plt.xticks(processors, processors)
plt.ylim(0, max(ideal_speedup) * 1.1)

plt.xlabel("Number of processors")
plt.ylabel("Speedup")
plt.title("Strong Scaling: Speedup Comparison")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.tight_layout()

# Save plot
plt.savefig("strong_scaling_speedup_512.png", dpi=300)
plt.close()  # Uncomment if you don’t want to display
#plt.show()
