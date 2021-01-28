import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Data files
directory = ''
fnames = ['learner_flockingAOI', 'baseline_random', 'baseline_mst']
fnames = [directory + fname + '.csv' for fname in fnames]
n_trials = 100

# Curve appearance
colors = ['tab:orange', 'tab:red', 'tab:green', 'tab:pink']
linestyles = ['-', '-.', '--']
labels = ['Non-Linear Agg. GNN', 'Random', 'MST']

# Generate plot
fig = plt.figure(figsize=(6, 4))

for fname, label, color, ls in zip(fnames, labels, colors, linestyles):
    import os 
    filename = os.path.join(R'C:\Users\Landon\Source\Repos\aoi_multi_agent_swarm\plot\flockingAOI', fname)
    data = np.loadtxt(fname, skiprows=1)
    plt.errorbar(data[:, 0], -1.0 * data[:, 1], yerr=data[:, 2] / np.sqrt(n_trials), label=label, color=color, ls=ls)

plt.ylabel('Avg. Cost')
plt.xlabel('Agent Velocity Ratio')
plt.legend(loc='lower right')
plt.ylim((9, 25))

# Save plot as .eps
# filename = os.path.join(R'C:\Users\Landon\Source\Repos\aoi_multi_agent_swarm\plot\flockingAOI', 'flocking_aoi.eps')
# filename = os.path.join(R'C:\Users\Landon\Source\Repos\aoi_multi_agent_swarm\plot\flockingAOI', 'flocking_aoi.eps')
plt.savefig('flocking_aoi.eps', format='eps', bbox_inches='tight')
plt.show()