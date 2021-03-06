import gym
import configparser
from os import path
import sys
import numpy as np
import aoi_envs
import csv


def eval_baseline(env, baseline, probability, n_episodes=20):
    """
    Evaluate a model against an environment over N games.
    """
    results = {'reward': np.zeros(n_episodes)}

    for k in range(n_episodes):
        done = False
        obs = env.reset()
        timestep = 1
        while not done:
            if baseline == 'MST':
                action = env.env.mst_controller(probability)
            elif baseline == 'Random':
                action = env.env.random_controller(probability)
            elif baseline == 'RoundRobin':
                action = env.env.roundrobin_controller()
            else:
                print('Not Baseline')

            obs, rewards, done, info = env.step(action)

            # Record results.
            results['reward'][k] += rewards
            timestep += 1

    mean_reward = np.mean(results['reward'])
    std_reward = np.std(results['reward'])
    # print(baseline + ' ' + str(probability) +  ', mean = {:.1f}, std = {:.1f}'.format(mean_reward, std_reward))
    return mean_reward, std_reward


def main(exp_name):

    if exp_name == 'power':
        environments = ['PowerLevel02Env-v0', 'PowerLevel025Env-v0', 'PowerLevel05Env-v0', 'PowerLevel075Env-v0', 'PowerLevel10Env-v0']
        filename = "power.csv"
    elif exp_name == 'mobile':
        environments = ['MobileEnv005-v0', 'MobileEnv01-v0', 'MobileEnv015-v0', 'MobileEnv025-v0', 'MobileEnv05-v0']
        filename = "mobile.csv"
    elif exp_name == 'flocking_aoi':
        environments = ['FlockingAOI015Env-v0', 'FlockingAOI025Env-v0', 'FlockingAOI0325Env-v0', 'FlockingAOI05Env-v0', 'FlockingAOI0625Env-v0', 'FlockingAOI075Env-v0']
        filename = "flocking_aoi.csv"
    elif exp_name == 'flocking':
        environments = ['Flocking015Env-v0', 'Flocking025Env-v0', 'Flocking0325Env-v0', 'Flocking05Env-v0', 'Flocking0625Env-v0', 'Flocking075Env-v0']
        filename = "flocking.csv"
    elif exp_name == 'mobile_n':
        environments = ['MobileEnv10N10-v0', 'MobileEnv10N20-v0', 'MobileEnv10N40-v0', 'MobileEnv10N60-v0', 'MobileEnv10N80-v0', 'MobileEnv10N100-v0']
        filename = 'mobile_n.csv'
    elif exp_name == 'n':
        environments = ['Stationary10Env-v0', 'Stationary20Env-v0', 'Stationary40Env-v0', 'Stationary60Env-v0', 'Stationary80Env-v0', 'Stationary100Env-v0']
        filename = 'n.csv'
    else:
        environments = ['StationaryEnv-v0']
        filename = "stationary.csv"

    baselines = ['Random', 'MST', 'RoundRobin']

    # probabilities = [0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]  #, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.5]
    probabilities = [0.04, 0.06, 0.08, 0.1, 0.12, 0.15]  #, 0.18, 0.2, 0.22, 0.25, 0.5]
    fields = ['EnvName']
    for i in baselines:
        fields.append(i + " Mean")
        fields.append(i + " Std")
        fields.append(i + " Prob")
    print(fields)

    data_to_csv = []
    for env_name in environments:
        best_results = [env_name]
        env = gym.make(env_name)
        print(env_name)
        for baseline in baselines:
            means = []
            
            if baseline == 'RoundRobin':
                best_prob = 0.0
            else:
                for p in probabilities:
                    m, _ = eval_baseline(env, baseline, p, n_episodes=50)
                    means.append(m)
                    print(m)
                max_ind = np.argmax(means)
                best_prob = probabilities[max_ind]

            final_mean, final_std = eval_baseline(env, baseline, best_prob, n_episodes=100)
            best_results.append(final_mean)
            best_results.append(final_std)
            best_results.append(best_prob)
        print(best_results)
        data_to_csv.append(best_results)
    
    # writing to csv file  
    with open(filename, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        
        # writing the fields  
        csvwriter.writerow(fields)  
        
        # writing the data rows  
        csvwriter.writerows(data_to_csv)


if __name__ == '__main__':
    main(sys.argv[1])
