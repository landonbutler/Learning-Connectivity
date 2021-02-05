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


def main():
    # params = ['N10', '', 'N40', 'N60', 'N80', 'N100']
    # env_names = ['MobileEnv10']
    # postfix = '-v0'
    #
    # environments = []
    # for env_name in env_names:
    #     for param in params:
    #         environments.append(env_name + param + postfix)

    environments = ['StationaryEnv-v0']
    # environments = ['MobileEnv10-v0']

    baselines = ['Random', 'MST']

    probabilities = [0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25]
    # probabilities = [0.08, 0.1, 0.12, 0.15]
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
            for p in probabilities:
                m, _ = eval_baseline(env, baseline, p, n_episodes=20)
                means.append(m)
                print(env.env.comm_model)
                print(m)
            max_ind = np.argmax(means)
            best_prob = probabilities[max_ind]
            final_mean, final_std = eval_baseline(env, baseline, best_prob, n_episodes=100)

            best_results.append(final_mean)
            best_results.append(final_std)
            best_results.append(best_prob)
        print(best_results)
        data_to_csv.append(best_results)

    filename = "tuned_push.csv"
    
    # writing to csv file  
    with open(filename, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        
        # writing the fields  
        csvwriter.writerow(fields)  
        
        # writing the data rows  
        csvwriter.writerows(data_to_csv)


if __name__ == '__main__':
    main()