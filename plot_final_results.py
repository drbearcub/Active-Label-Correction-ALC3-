import json
import glob
import matplotlib.pyplot as plt
import re

# Collect all files matching your naming pattern
files = sorted(glob.glob("alcIterations/iteration_*_step_2_probabilities.json"))


# Extract iteration numbers and sort numerically
def extract_iteration(filename):
    match = re.search(r"iteration_(\d+)_step_2_probabilities\.json", filename)
    return int(match.group(1)) if match else -1


files = sorted(files, key=extract_iteration)

avg_geo_means = []
avg_forced_geo_means = []
iterations = []

for file in files:
    if file == "alcIterations/iteration_7_step_2_probabilities.json":
        break
    with open(file, "r") as f:
        data = json.load(f)  # each file contains a list of dictionaries

    geo_means = [entry["geo_mean"] for entry in data if "geo_mean" in entry]
    forced_geo_means = [entry["forced_geo_mean"] for entry in data if "forced_geo_mean" in entry]

    avg_geo = sum(geo_means) / len(geo_means) if geo_means else 0
    avg_forced_geo = sum(forced_geo_means) / len(forced_geo_means) if forced_geo_means else 0

    avg_geo_means.append(avg_geo)
    avg_forced_geo_means.append(avg_forced_geo)

    iterations.append(extract_iteration(file))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(iterations, avg_geo_means, marker="o", label="Average geo_mean")
plt.plot(iterations, avg_forced_geo_means, marker="x", label="Average forced_geo_mean")
plt.xlabel("Iteration")
plt.ylabel("Average Value")
plt.title("Average geo_mean vs forced_geo_mean per iteration")
plt.legend()
plt.grid(True)

# Save to file
plt.savefig("geo_mean_plot.png", dpi=300)
print("✅ Plot saved as geo_mean_plot.png")
