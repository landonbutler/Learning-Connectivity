import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Data files
directory = ''
fnames = ['id', 'nl', 'baseline_rr', 'baseline_mst']
fnames = [directory + fname + '.csv' for fname in fnames]
n_trials = 100

# Curve appearance
colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:pink']
linestyles = ['-', '-', '-.', '--']
labels = ['Agg. GNN', 'Non-Linear Agg. GNN', 'Round Robin', 'MST']

# Generate plot
fig = plt.figure(figsize=(6, 4))
for fname, label, color, ls in zip(fnames, labels, colors, linestyles):
    data = np.loadtxt(fname, skiprows=1)
    plt.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2] / np.sqrt(n_trials), label=label, color=color, ls=ls)

plt.ylabel('Avg. Reward')
plt.xlabel('GNN Receptive Field')
plt.legend(loc='lower right')
plt.ylim((-25, -9))

# Save plot as .eps
plt.savefig(directory + 'hops.eps', format='eps')
plt.show()

