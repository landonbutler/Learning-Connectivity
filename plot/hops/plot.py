import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Data files
directory = ''
fnames = ['id', 'nl', 'baseline_rr', 'baseline_mst', 'baseline_random', 'baseline_random_flooding']
fnames = [directory + fname + '.csv' for fname in fnames]
n_trials = 100

# Curve appearance
colors = ['tab:blue', 'tab:orange', 'tab:purple', 'tab:green', 'tab:red', 'tab:pink']
linestyles = [':', '-', '-.', '--', ':', ':']
labels = ['Agg. GNN', 'Non-Linear Agg. GNN', 'Round Robin', 'MST', 'Random', 'Random Flooding']

# Generate plot
fig = plt.figure(figsize=(6, 4))
plt.tight_layout()
for fname, label, color, ls in zip(fnames, labels, colors, linestyles):
    data = np.loadtxt(fname, skiprows=1)
    plt.errorbar(data[:, 0], -1.0 * data[:, 1], yerr=data[:, 2] / np.sqrt(n_trials), label=label, color=color, ls=ls)

plt.ylabel('Avg. Cost')
plt.xlabel('GNN Receptive Field')
plt.legend(loc='upper right')
plt.ylim((9, 25))

# Save plot as .eps
plt.savefig(directory + 'hops.eps', format='eps', bbox_inches='tight')
plt.show()


