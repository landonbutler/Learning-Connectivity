import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Data files
directory = ''
fnames = ['id', 'nl', 'mst', 'random']
fnames = [directory + fname + '.csv' for fname in fnames]
n_trials = 100

# Curve appearance
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
linestyles = ['--', '-', '-.', ':']
labels = ['Agg. GNN', 'Non-Linear Agg. GNN', 'MST', 'Random']

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


