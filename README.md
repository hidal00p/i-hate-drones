# `i-hate-drones`

Piece of software that provides a workflow for training RL-based drone agents to fly autonomously in a simulated environement.

## Software stack

This pakage is pretty much a wrapper around the following projects which deliver
the horsepower for the RL workflow:

- [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) - provides a training environment for the flight by interfacing with the pybullet physics engine.
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- torch - for neural nets and optimizers.
- logging and visualization to be defined.
- probably some db(sqite) for training snapshots persistence.

## Build

tbd...

## Usage

tbd... not really sure how to define user experience yet.

## Project structure

Once the project matures a little add the file structure.