# Hierarchical policy for rai

This repo contains the python code for a hierarchical policy for the RAI code in python bindings. See https://github.com/HM-Van/rai for a README of the modified RAI code. See https://github.com/MarcToussaint/rai-python for the original python bindings.

This repo is currently integrated in https://github.com/HM-Van/rai-python/. Follow the instructions to create the conda environment and for the installation.


## Create data set

In order to collect training samples for the environment with the id nenv, the file "./models/Test\_setup\_nenv.g" has to exist.

```
python3 database.py --env=nenv
 
```

The training samples can then be combined into a data set:
```
# use data set expansion for 7 predefined objectives. skip1 flag only trains objectives consisting of two goal formulations. If rand2>7, radomly selected objectives are added
python3 database.py --mixData -- rand2=0 --skip1

# Exclude certain goal formulations and their combinations (here hardcoded: (on red table1))
python3 database.py --exclude

#  If rand2=0, select 40 hardcoded objectives. I rand2>0, select randomly selected objectives
python3 database.py -- rand2=0 --skip1

```

## Train and use hierarchical policy

In order to train a hierarchical policy:
```
# model\_dir\_data is timestamp only (without _final). NNmode can be "minimal" (simple FF NN), "FFnew" (Classifier chain), "mixed10" (LSTM)
# datasetMode: 1(global coord) 2(relative coord) 3(global coord+encoder) 4(relative coord+encoder)
# Additional training parameters can be added, see main.py
python3 main.py --saveModel --train\_only --model\_dir\_data="20200220" --NNmode="FFnew" --datasetMode=1

```

In order to use a trained hierarchical policy (the file "./models/Test\_setup\_nenv.g" has to exist):
```
# model\_dir is timestamp only (without e.g. _FFnew1)
# --exlude: test certain goal formulations and their combinations (here hardcoded: (on red table1))
# Additional testing parameters can be added, see main.py

# Test all objectives
# allEnv tests a sequence of environments starting from env
python3 main.py --model\_dir="2020002021-152316" --NNmode="FFnew" --datasetMode=1 --completeTesting --env=100 --allEnv

# Test single objective
# goal formulations (held object), (on object table)
python3 main.py --model\_dir="2020002021-152316" --NNmode="FFnew" --datasetMode=1 --goal="(on red table1) (on blue table2)" --env=100

```

## Use original framework

In order to use the original LGP solution (the file "./models/Test\_setup\_nenv.g" has to exist):
```
# goal formulations (held object), (on table object)
# !! note that the order for (on table object) is reversed compared to above!!
python3 rai\_skeleton.py --env=100 --goal="(on table1 red) (on table2  blue)"

```


