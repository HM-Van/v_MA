# Hierarchical policy for rai

This repo contains the python code for a hierarchical policy for the RAI code in python bindings. See https://github.com/HM-Van/rai for a README of the modified RAI code. See https://github.com/MarcToussaint/rai-python for the original python bindings.

This repo is currently integrated in https://github.com/HM-Van/rai-python/. Follow the instructions to create the conda environment and for the installation.


## Create data set

In order to collect training samples for the environment with the id nenv, the file "./models/Test_setup_nenv.g" has to exist.

```
# --NNmode: "final" (initial setup) or "stack" (modified setup)
# --env: enironment to test

python3 database.py --env=nenv --NNmode="final"
 
```

The training samples can then be combined into a data set:
```
# --NNmode can be "final" (initial setup), "stack" (modified setup)

#----------------------------------------------------------------
# --mixData for data set expansion for 7 predefined objectives
# If --rand2>7, radomly selected objectives are added
# --skip1: only trains objectives consisting of two goal formulations.

python3 database.py --mixData -- rand2=0 --skip1 --NNmode="final"

#----------------------------------------------------------------
# Exclude certain goal formulations and their combinations (here hardcoded: (on red table1))

python3 database.py --exclude --NNmode="final"

#----------------------------------------------------------------
# Combine objectives without data set expansion
# If --rand2=0, select 40 hardcoded objectives. If --rand2>0, select randomly rand2 objectives

python3 database.py -- rand2=0 --skip1 --NNmode="final"

```

## Train and use hierarchical policy

In order to train a hierarchical policy (the file "./models/Test_setup_nenv.g" has to exist):
```
# --model_dir_data is timestamp only (without _final)
# --NNmode can be "minimal" (simple FF NN), "FFnew" (Classifier chain), "mixed10" (LSTM)
# --datasetMode for initial setup: 1(global coord) 2(relative coord) 3(global coord+encoder) 4(relative coord+encoder)
# --datasetMode fot modified setup: 5(global coord) 6(relative coord) 7(global coord+encoder) 8(relative coord+encoder)
# --env: required to obtain symbols of the environment, so that inputsize is calculated correctly
# --saveModel: trains a model for trainingset in model_dir_data

python3 main.py --saveModel --train_only --model_dir_data="20200220" --NNmode="FFnew" --datasetMode=1 --env=100

#----------------------------------------------------------------
# Additional training parameters can be added, see main.py, e.g.
# --train_only: model is only trained and not tested for specific objectives
# --epochs_inst: number of epochs for training
# --size_inst: number of neurons per hidden layer
# --hlayers_inst: number of hidden layers
# --hlayers_inst2: number of hidden LSTM layers (for implementation3 and instruction network only)
# above are for instruction network, replace _inst with _grasp or _place for respective networks

```

In order to use a trained hierarchical policy (the file "./models/Test_setup_nenv.g" has to exist):
```
# model_dir is timestamp only (without e.g. _FFnew1)
# --NNmode can be "minimal" (simple FF NN), "FFnew" (Classifier chain), "mixed10" (LSTM)
# --datasetMode for initial setup: 1(global coord) 2(relative coord) 3(global coord+encoder) 4(relative coord+encoder)
# --datasetMode fot modified setup: 5(global coord) 6(relative coord) 7(global coord+encoder) 8(relative coord+encoder)
# --env: env to evaluate

#----------------------------------------------------------------
# Test all objectives
# --completeTesting: tests all objectives
# --allEnv: tests a sequence of environments starting from env

python3 main.py --model_dir="2020002021-152316" --NNmode="FFnew" --datasetMode=1 --completeTesting --env=100 --allEnv

#----------------------------------------------------------------
# Test single objective
# goal formulations (held object), (on object table)
# --goal: sequence of goal formulations

python3 main.py --model_dir="2020002021-152316" --NNmode="FFnew" --datasetMode=1 --goal="(on red table1) (on blue table2)" --env=100

#----------------------------------------------------------------
# Additional testing parameters can be added, see main.py, e.g.
# --exlude: test certain goal formulations and their combinations (here hardcoded: (on red table1))
# --cheat_goalstate: objective gets adapted to only consist of unsatisfied goal formulations
# --tray="tray" : objective gets expanded to place object on table with name "tray" (or on its target) early, if object below has to be placed on it
# --newLGP: reloads lgp at intermediate node in tree. Can reduce time of optimization for next configuration, but position of previously placed objects can get shifted
# --maxDepth: maximum depth of tree = maximum length of skeleton before reattempting search
# --maxTries: maximum number of tries to find skeleton
# --maxTimeOP: time for final path optimization. As long as this time is not exceeded, optimization is restarted with random noise added to initial configuration
# --planOnly: currently set automatically. Next configuration computed through seq bound instead of seqPath bound
# --viewConfig: displays view of environment
# --showFinal: displays computed path after final path optimization

```

## Use original framework

In order to use the original LGP solution (the file "./models/Test_setup_nenv.g" has to exist):
```
# goal formulations (held object), (on table object)
# !! note that the order for (on table object) is reversed compared to above!!
# Currently does not stop automatically. Either stops once memory limit is reached or after ctrl+c

python3 rai_skeleton.py --env=100 --goal="(on table1 red) (on table2  blue)"

```


