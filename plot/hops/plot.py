import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Data files
directory = ''
# fnames = ['id', 'nl', 'mst', 'random']
fnames = ['nl', 'mst', 'random', 'rr']
fnames = [directory + fname + '.csv' for fname in fnames]
n_trials = 100

# colors = ['tab:orange', 'tab:green', 'tab:red']
# fnames = ['nl20', 'nl60', 'rr', 'mst', 'random']
colors = ['tab:orange','tab:purple', 'tab:green', 'tab:pink']
linestyles = ['-', '-.', '--', 'dotted']
labels = ['GNN', 'MST', 'Random', 'Round Robin']

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
# plt.show()


