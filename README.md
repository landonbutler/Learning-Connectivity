# Learning Connectivity for Data Distribution in Robot Teams
<p align="center">
  <a href="https://www.youtube.com/watch?v=UNBvsPZIudU">
<img src="https://github.com/landonbutler/Learning-Connectivity/blob/main/visuals/MST-Visual.gif" width="700">
    <a\>
</p>
  
## Set up

To set this project up to work with Docker, see the `docker/` directory for the Dockerfile and a bash script to start the container.

## Dependencies
* graph_nets
```shell
pip install graph_nets "tensorflow>=1.15,<2" "dm-sonnet<2" "tensorflow_probability<0.9"
```
* gym==0.11.0

* progress

* stable_baselines==2.9.0

## Model Training

```python3 train.py -p cfg/nl.cfg```
where the first argument is the path of the experiment config file.

## Model Testing

```python3 test_all.py cfg/nl.cfg hops```
where the first argument is the config file that was used for training and the second argument is the filename for the output results.

## Visualization

```python3 test.py -p <path to model> -e StationaryEnv-v0 -v```
add -gif to save visualization to a GIF

## Configuration files
The following configuration files were used to train the models for the paper:
* nl.cfg - varying the number of hops in the GNN
* n.cfg - stationary agents, varying the team size
* power.cfg - stationary agents, varying the transmit power
* flocking.cfg - varying the flocking initial velocity (variance of velocities cost) 
* flocking_aoi.cfg - varying the flocking initial velocity (Age of Information cost)
* mobile.cfg - random walk mobile agents, varying the agent velocity
* n_mobile.cfg - random walk mobile agents, varying the team size

The following additional configuration files are only used for testing:
* gen_flocking_aoi_to_var.cfg
* gen_flocking_var_to_aoi.cfg
* gen_n.cfg
* gen_n_mobile.cfg
