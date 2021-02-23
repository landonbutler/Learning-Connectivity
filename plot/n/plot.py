import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Data files
directory = ''
fnames = ['train', 'gen', 'rr', 'mst', 'random']
fnames = [directory + fname + '.csv' for fname in fnames]
n_trials = 1000
gnn_n_trials = 100

# Curve appearance
colors = ['tab:orange','tab:blue', 'tab:purple', 'tab:green', 'tab:pink']
linestyles = ['-', '-', '-.', '--', 'dotted']
labels = ['GNN Trained','GNN Generalization', 'Round Robin', 'MST', 'Random Flooding']

# Generate plot
fig = plt.figure(figsize=(6, 4))
# plt.tight_layout()
for fname, label, color, ls in zip(fnames, labels, colors, linestyles):
    data = np.loadtxt(fname, skiprows=1)
    trials = 0
    if fname is "nl":
        trials = gnn_n_trials
    else:
        trials = n_trials
    plt.errorbar(data[:, 0], -1.0 * data[:, 1], yerr=data[:, 2] / np.sqrt(trials), label=label, color=color, ls=ls)

plt.ylabel('Avg. Cost')
plt.xlabel('Number of Agents')
plt.legend(loc='lower right')
# plt.ylim((-25, -9))

# Save plot as .eps
plt.savefig(directory + 'n.eps', format='eps', bbox_inches='tight')
plt.savefig(directory + 'n.png', format='png', bbox_inches='tight')
plt.show()


