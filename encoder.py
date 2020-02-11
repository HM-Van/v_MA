import minimal_experiment as expert

import tensorflow as tf
from tensorflow import keras
import numpy as np
import rai_world

import sys
import os
dir_file=os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir_file+'/../ry/')
from libry import *

import datetime

# From https://blog.keras.io/building-autoencoders-in-keras.html
rai=rai_world.RaiWorld(dir_file, 1, "minimal", "", 0, maxDepth=20)

if True: #Goalencoder
    datasetIn=np.zeros((72*2+15,20), dtype=int)
    datasetOut=np.zeros((72*2+15,20), dtype=int)
    for i in range(0,72*2,2):
        goalStep = rai_world.splitStringStep(expert.Sets[i], list_old=[],verbose=0)
        goalState= rai.encodeGoal(goalStep)
        goalStep1 = rai_world.splitStringStep(expert.Sets[i+1], list_old=[],verbose=0)
        goalState1= rai.encodeGoal(goalStep1)

        datasetIn[i,:] = goalState
        datasetIn[i+1,:] = goalState1
        datasetOut[i,:] = goalState
        datasetOut[i+1,:] = goalState

    for i in range(15):
        goalState= rai.encodeGoal([expert.test[i], expert.test[i]])
        datasetIn[2*72+i,:] = goalState
        datasetOut[2*72+i,:] = goalState
    np.set_printoptions(threshold=sys.maxsize)
    input0=keras.Input(shape=(20,))

    encoded = keras.layers.Dense(16, activation=keras.activations.relu, kernel_regularizer=keras.regularizers.l2(0.0001))(input0)
    decoded = keras.layers.Dense(20, activation=keras.activations.sigmoid)(encoded)

    autoencoder=keras.models.Model(inputs=input0, outputs=decoded)
    encoder=keras.models.Model(inputs=input0, outputs=encoded)

    encodedInput=input0=keras.Input(shape=(16,))
    decodeLayer=autoencoder.layers[-1]
    decoder=keras.models.Model(inputs=encodedInput, outputs=decodeLayer(encodedInput))

    autoencoder.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss="binary_crossentropy") #decay=0.001/self.epochs_inst
    autoencoder.fit(datasetIn, datasetOut, epochs=800, batch_size=32, shuffle=True)

    encoded_test=encoder.predict(datasetIn)
    decoded_test=decoder.predict(encoded_test)

    #print(decoded_test)

    autoencoder.save(dir_file+'/logs/encoder/autoencoderGoal2.h5')
    encoder.save(dir_file+'/logs/encoder/encoderGoal2.h5')
    decoder.save(dir_file+'/logs/encoder/decoderGoal2.h5')