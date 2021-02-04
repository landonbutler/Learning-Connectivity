import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Data files
directory = ''
fnames = ['nl20', 'nl60', 'random', 'mst']
fnames = [directory + fname + '.csv' for fname in fnames]
n_trials = 100

# Curve appearance
colors = ['tab:orange', 'tab:blue', 'tab:red', 'tab:green', ]
linestyles = ['-', '-', '-.', '--']
labels = ['GNN Trained on 20 Agents', 'GNN Trained on 60 Agents', 'Random', 'MST']

# Generate plot
fig = plt.figure(figsize=(6, 4))
# plt.tight_layout()
for fname, label, color, ls in zip(fnames, labels, colors, linestyles):
    data = np.loadtxt(fname, skiprows=1)
    plt.errorbar(data[:, 0], -1.0 * data[:, 1], yerr=data[:, 2] / np.sqrt(n_trials), label=label, color=color, ls=ls)

plt.ylabel('Avg. Cost')
plt.xlabel('Number of Agents')
plt.legend(loc='lower right')
# plt.ylim((-25, -9))

# Save plot as .eps
plt.savefig(directory + 'n.eps', format='eps', bbox_inches='tight')
plt.show()


