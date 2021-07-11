import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Data files
directory = ''
fnames = ['nl', 'random', 'mst']
fnames = [directory + fname + '.csv' for fname in fnames]
n_trials = 100

# Curve appearance
colors = ['tab:orange', 'tab:red', 'tab:green']
linestyles = ['-', '-.', '--']
labels = ['Non-Linear Agg. GNN', 'Random', 'MST']


# Generate plot
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)
plt.tight_layout()
for fname, label, color, ls in zip(fnames, labels, colors, linestyles):
    data = np.loadtxt(fname, skiprows=1)
    plt.errorbar(data[:, 0], -1.0 * data[:, 1], yerr=data[:, 2] / np.sqrt(n_trials), label=label, color=color, ls=ls)

# ax.set_yscale('log')
plt.ylabel('Avg. Age of Info. Cost')
plt.xlabel('Agent Velocity Ratio')
plt.legend(loc='upper right')

# plt.ylim((-25, -9))

# Save plot as .eps
plt.savefig(directory + 'mobile.eps', format='eps', bbox_inches='tight')
plt.show()


