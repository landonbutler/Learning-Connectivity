import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

for reward in ['aoi', 'var']:
    # Data files
    directory = reward + '/'
    fnames = ['test_aoitrain_nl', 'test_vartrain_nl', '_baseline_random', '_baseline_mst']
    fnames = [directory + reward + fname + '.csv' for fname in fnames]
    n_trials = 100

    # Curve appearance
    colors = ['tab:orange', 'tab:pink', 'tab:red', 'tab:green']
    linestyles = ['-', '-', '-.', '--']
    labels = ['Agg. GNN Trained using AoI', 'Agg. GNN Trained using Variance', 'Random', 'MST']

    # Generate plot
    fig = plt.figure(figsize=(6, 4))

    for fname, label, color, ls in zip(fnames, labels, colors, linestyles):
        # import os
        # filename = os.path.join(R'C:\Users\Landon\Source\Repos\aoi_multi_agent_swarm\plot\flocking', fname)
        data = np.loadtxt(fname, skiprows=1)
        plt.errorbar(data[:, 0], -1.0 * data[:, 1], yerr=data[:, 2] / np.sqrt(n_trials), label=label, color=color, ls=ls)

    plt.ylabel('Avg. Cost')
    plt.xlabel('Agent Velocity Ratio')
    plt.legend(loc='lower right')

    if reward == 'aoi':
        plt.ylim((9, 25))
    else:
        plt.ylim((0, 50))

    # Save plot as .eps
    # filename = os.path.join(R'C:\Users\Landon\Source\Repos\aoi_multi_agent_swarm\plot\flocking', 'flocking_aoi.eps')
    plt.savefig('flocking_' + reward + '.eps', format='eps', bbox_inches='tight')
    # plt.show()