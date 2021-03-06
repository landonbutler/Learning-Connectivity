import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Data files
directory = ''
# fnames = ['id', 'nl', 'mst', 'random']
fnames = ['nl40', 'rr40', 'mst40', 'random40']
fnames = [directory + fname + '.csv' for fname in fnames]
n_trials = 1000

# Curve appearance
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
# linestyles = ['--', '-', '-.', ':']
# labels = ['Agg. GNN', 'Non-Linear Agg. GNN', 'MST', 'Random']

colors = ['tab:orange', 'tab:purple', 'tab:green', 'tab:pink']
linestyles = ['-', '-.', '--', 'dotted']
labels = ['Non-Linear Agg. GNN', 'Round Robin', 'MST', 'Random Flooding']

# Generate plot
fig = plt.figure(figsize=(6, 4))
plt.tight_layout()
for fname, label, color, ls in zip(fnames, labels, colors, linestyles):
    data = np.loadtxt(fname, skiprows=1)
    plt.errorbar(data[:, 0], -1.0 * data[:, 1], yerr=data[:, 2] / np.sqrt(n_trials), label=label, color=color, ls=ls)

plt.ylabel('Avg. Cost')
plt.xlabel('GNN Receptive Field')
plt.legend(loc='upper right')
# plt.ylim((25, 50))

# Save plot as .eps
plt.savefig(directory + 'hops40.eps', format='eps', bbox_inches='tight')
plt.show()


