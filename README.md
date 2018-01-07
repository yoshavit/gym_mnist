# MNIST Games
This is the code for the environment benchmarks used in 
"[Learning Environment Simulators from Sparse Signals](http://yonadavshavit.com/assets/files/masters-engineering-thesis.pdf)" (Shavit 2017).

To use the environments defined in the package, navigate to the root of this project's directory and call:
```bash
python setup.py install
```
From now on, you can `import gym_mnist` and then use gym's `gym.make()` to construct the MNIST-game envs.

To interactively play the different environments, call
```bash
python test-env.py [environment_name]
```

For a list of the environment names, simply call 
```bash
python test-env.py -h
```
