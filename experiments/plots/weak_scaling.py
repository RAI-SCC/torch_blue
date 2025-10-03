import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your own)
processors = np.array([1, 2, 4, 8, 16])
times = np.array([48.04, 48.42, 48.46, 48.62, 49.57])  # runtimes for weak scaling

# Compute efficiency
efficiency = times[0] / times  

# Plot
plt.figure(figsize=(7, 5))
plt.plot(processors, efficiency, marker='o', linestyle='-', label="Weak scaling efficiency")

# Ideal efficiency line
plt.axhline(1.0, color='gray', linestyle='--', linewidth=1, label="Ideal")

# Format plot
plt.xscale("log", base=2)   # optional: log scale for processors
plt.xticks(processors, processors)
plt.ylim(0, 1.1)

plt.xlabel("Number of processors")
plt.ylabel("Efficiency")
plt.title("Weak Scaling Efficiency")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.tight_layout()

# Save as PNG
plt.savefig("weak_scaling_efficiency.png", dpi=300)

# If you don’t want the plot window to appear:
plt.close()
