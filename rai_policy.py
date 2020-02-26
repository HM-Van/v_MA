import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import Model
from keras.objectives import categorical_crossentropy

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as KB

import math
import datetime
import shutil
import os
import random

# Old policy file => see policy2.py

def build_Sub(inputs, 
              output_size,
              scope,
              name,
              n_layers=2, 
              size=200, 
              activation=keras.activations.relu,
              output_activation=keras.activations.softmax,
              reg=0.001
              ):
    out = inputs
    with tf.variable_scope(scope):
        for i in range(n_layers):
            out = keras.layers.Dense(size, activation=activation, kernel_regularizer=keras.regularizers.l2(reg))(out)
            if i<(n_layers-1):
                out= keras.layers.Dropout(0.2)(out)
        out = keras.layers.Dense(output_size, activation=output_activation, name=name)(out)
    return out

def build_LSTM(inputs, 
              output_size,
              scope,
              name,
              n_layers=1, 
              size=200, 
              activation=keras.activations.relu,
              output_activation=keras.activations.softmax,
              reg=0.0
              ):
    out = inputs # input (batch_size, timespan, input_dim)
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = keras.layers.LSTM(size, activation=activation, dropout=0.2, return_sequences=True, kernel_regularizer=keras.regularizers.l2(reg), recurrent_regularizer=keras.regularizers.l2(reg), bias_regularizer=keras.regularizers.l2(reg))(out) # output (batch_size, timespan, size)
        final = keras.layers.LSTM(output_size, activation=output_activation, return_sequences=False, kernel_regularizer=keras.regularizers.l2(reg), recurrent_regularizer=keras.regularizers.l2(reg), bias_regularizer=keras.regularizers.l2(reg), name=name)(out) # output (batch_size, timespan, output_size)

    return out, final

class printEpoch(tf.keras.callbacks.Callback): 
	def on_epoch_end(self, epoch, logs={}):
		if (epoch%10)==0:
			now=datetime.datetime.now()
			timestamp=str(now.year)+"."+str(now.month).zfill(2)+"."+str(now.day).zfill(2)+" "+str(now.hour).zfill(2)+":"+str(now.minute).zfill(2)+":"+str(now.second).zfill(2)
			print("Epoch "+str(epoch).rjust(5)+" ended on "+timestamp+", loss: "+"{:.6f}".format(logs.get('loss'))+", val_loss: "+"{:.6f}".format(logs.get('val_loss')))

def EarlyStopping(val_split, patience=50):
    if val_split>0.0:
        return keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=patience, mode='min') #, restore_best_weights=True
    else:
        return keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=patience, mode='min')

def estimateT(epochs, num=1):
    time=int(num*45*epochs/1000)
    return "Estimated Time: "+str(time//60)+"h "+str(time%60)+"min"

def estimateT_lstm(epochs, num=1):
    time=int(num*10*epochs/100)
    return "Estimated Time: "+str(time//60)+"h "+str(time%60)+"min"


class FeedForwardNN():
    def __init__(self): #no itialization -> either load or build
        pass

    #----------Load model from file-----------------
    def load_net(self,path_rai, model_dir=''):
        if not model_dir=='':
            model_dir=model_dir+'_minimal/'
        self.modelInstruct=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelInstruct.h5')
        self.modelGrasp=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelGrasp.h5')
        self.modelPlace=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelPlace.h5')

    #----------Build model------------
    def build_net(self,
                inputlength,
                numInstruct,
                numGripper,
                numObjects,
                numTables,
                epochs_inst=300,
                n_layers_inst=2, 
                size_inst=200,
                epochs_grasp=300,
                n_layers_grasp=2,
                size_grasp=200,
                epochs_place=300,
                n_layers_place=2,
                size_place=200,
                timestamp="",
                lr=0.001,
                lr_drop=0.5,
                epoch_drop=100,
                val_split=0.2,
                mode=1,
                listLog=[]
                ):

        self.inputlength=inputlength
        self.numInstruct=numInstruct
        self.numGripper=numGripper
        self.numObjects=numObjects
        self.numTables=numTables

        self.epochs_inst=epochs_inst
        self.hlayers_inst=n_layers_inst
        self.size_inst=size_inst

        self.epochs_grasp=epochs_grasp
        self.hlayers_grasp=n_layers_grasp
        if self.hlayers_grasp<1:
            self.hlayers_grasp=1
        self.size_grasp=size_grasp

        self.epochs_place=epochs_place
        self.hlayers_place=n_layers_place
        if self.hlayers_place<1:
            self.hlayers_place=1
        self.size_place=size_place

        self.lr=lr
        self.lr_drop=lr_drop
        self.epoch_drop=epoch_drop
        self.val_split=val_split
        self.mode=mode

        self.params=[]
        param=["learning rate"]
        param.append("init: "+str(self.lr))
        param.append("drop: "+str(self.lr_drop))
        param.append("epoch_drop: "+str(self.epoch_drop))
        param.append("val_split: "+str(self.val_split))
        self.params.append(param)

        self.listLog=listLog
        if self.listLog is []:
            self.listLog.append([0,1])
            self.listLog.append([2,3,4])
            self.listLog.append([2,3,4,5,6])

        print(self.listLog)
        
        if timestamp=="":
            now=datetime.datetime.now()
            self.timestamp=str(now.year)+str(now.month).zfill(2)+str(now.day).zfill(2)+"-"+str(now.hour).zfill(2)+str(now.minute).zfill(2)+str(now.second).zfill(2)+"_minimal"
        else:
            self.timestamp=timestamp

        self.modelInstruct = self.build_Instruct()
        self.modelGrasp    = self.build_Grasp()
        self.modelPlace    = self.build_Place()

        lossesInstruct={"instructOut": "categorical_crossentropy"}
        lossesGrasp={"graspGripperOut": "categorical_crossentropy", "graspObjectOut": "categorical_crossentropy"}
        lossesPlace={"placeGripperOut": "categorical_crossentropy", "placeObjectOut": "categorical_crossentropy", "placeTableOut": "categorical_crossentropy"}

        weightsInstruct={"instructOut": 1.0}
        weightsGrasp={"graspGripperOut": 1.0, "graspObjectOut": 1.0}
        weightsPlace={"placeGripperOut": 1.0, "placeObjectOut": 1.0, "placeTableOut": 1.0}

        self.modelInstruct.compile(optimizer=keras.optimizers.Adam(lr=self.lr), loss=lossesInstruct, loss_weights=weightsInstruct) #decay=0.001/self.epochs_inst
        self.modelGrasp.compile(optimizer=keras.optimizers.Adam(lr=self.lr), loss=lossesGrasp, loss_weights=weightsGrasp)#decay=0.001/self.epochs_grasp
        self.modelPlace.compile(optimizer=keras.optimizers.Adam(lr=self.lr), loss=lossesPlace, loss_weights=weightsPlace)#decay=0.001/self.epochs_place

    #------------------build subnetwork---------------
    def build_Instruct(self):
        # One Hot : Multi label classification
        param=["Instruct"]
        param.append("hlayers_inst: "+str(self.hlayers_inst))
        param.append("size_inst: "+str(self.size_inst))
        param.append("epochs_inst: "+str(self.epochs_inst))
        self.params.append(param)

        inputs = keras.Input(shape=(self.inputlength,))
        InstructNet = build_Sub(inputs, self.numInstruct, "instructScope", "instructOut", n_layers=self.hlayers_inst, size=self.size_inst)
        
        modelInstuct = keras.models.Model(inputs=inputs, outputs=[InstructNet], name="instructionNet")
        return modelInstuct

    def build_Grasp(self):
        # Multi Hot: Multi class classification
        param=["Grasp"]
        param.append("hlayers_grasp: "+str(self.hlayers_grasp))
        param.append("size_grasp: "+str(self.size_grasp))
        param.append("epochs_grasp: "+str(self.epochs_grasp))
        self.params.append(param)

        inputs = keras.Input(shape=(self.inputlength,))
        GripperNet = build_Sub(inputs, self.numGripper, "graspGripperScope", "graspGripperOut", n_layers=self.hlayers_grasp, size=self.size_grasp)
        ObjNet = build_Sub(inputs, self.numObjects, "graspObjectScope", "graspObjectOut", n_layers=self.hlayers_grasp, size=self.size_grasp)

        modelGrasp = keras.models.Model(inputs=inputs, outputs=[GripperNet, ObjNet], name="graspNet")
        return modelGrasp

    def build_Place(self):
        # Multi Hot: Multi class classification
        param=["Place"]
        param.append("hlayers_place: "+str(self.hlayers_place))
        param.append("size_place: "+str(self.size_place))
        param.append("epochs_place: "+str(self.epochs_place))
        self.params.append(param)

        inputs = keras.Input(shape=(self.inputlength,))
        GripperNet = build_Sub(inputs, self.numGripper, "placeGripperScope", "placeGripperOut", n_layers=self.hlayers_place, size=self.size_place)
        ObjNet = build_Sub(inputs, self.numObjects, "placeObjectScope", "placeObjectOut", n_layers=self.hlayers_place, size=self.size_place)
        TableNet = build_Sub(inputs, self.numTables, "placeTableScope", "placeTableOut", n_layers=self.hlayers_place, size=self.size_place)

        modelPlace = keras.models.Model(inputs=inputs, outputs=[GripperNet, ObjNet, TableNet], name="placeNet")
        return modelPlace

    #------------------Train and save Model---------------

    def step_decay(self,epoch):
        initial_lrate = self.lr
        drop = self.lr_drop
        epochs_drop = self.epoch_drop
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate

    def reshapeInput(self, path_rai, model_dir):

        if self.mode in [11,12]:
            Dappend="_new"
        else:
            Dappend=""

        if self.mode == 12:
            NPappend="_feat"
        else:
            NPappend=""

        dataInput=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'Input'+NPappend+'.npy') #None inputsize
        dataInstruct=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'Instruction.npy') #None numinstr
        dataLogicals=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'Logicals.npy') #None 3 numlog

        idx=list(range(dataInput.shape[0]))
        random.shuffle(idx)

        dataInput=dataInput[idx,:]
        dataInstruct=dataInstruct[idx,:]
        dataLogicals=dataLogicals[idx,:]

        #-------------------------------------------
        #----grasp gripper object

        graspIn=dataInput[dataInstruct[:,0]==1,:]
        graspFinalOut=dataLogicals[dataInstruct[:,0]==1,:].astype('int')

        #-------------------------------------------
        #----place gripper object table

        placeIn=dataInput[dataInstruct[:,1]==1,:]
        placeFinalOut=dataLogicals[dataInstruct[:,1]==1,:].astype('int')

        return dataInput, dataInstruct, graspIn, graspFinalOut, placeIn, placeFinalOut

    def train(self, path_rai, model_dir, saveToFile=True):
        if not model_dir=='':
            Dappend=""
            Setappend=""
            if self.mode==1:
                model_dir=model_dir+'/'
            elif self.mode==2:
                model_dir=model_dir+'_mixed2/'
            elif self.mode==0:
                model_dir=model_dir+'_mixed0/'
            elif self.mode in [11,12]:
                model_dir=model_dir+"_final/"
                Dappend="_new"
                Setappend="_"+str(self.mode)

        dataInput, dataInstruct, graspFinalIn, graspFinalOut, placeFinalIn, placeFinalOut=self.reshapeInput(path_rai, model_dir)


        tf.keras.backend.get_session().run(tf.global_variables_initializer())

        tbInstruction = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/Instruction')
        tbGrasp = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/Grasp')
        tbPlace = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/Place')

        if not(os.path.exists(path_rai+'/logs/'+self.timestamp+'/tmp')):
                os.makedirs(path_rai+'/logs/'+self.timestamp+'/tmp')

        saveInst=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelInstruct.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        savegrasp=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelGrasp.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        saveplace=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelPlace.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)

        if saveToFile:
            if not(os.path.exists(path_rai+'/logs/'+self.timestamp)):
                os.makedirs(path_rai+'/logs/'+self.timestamp)

            with open(path_rai+'/logs/'+self.timestamp+'/params.txt', 'w') as f:
                for item in self.params:
                    for i in item:
                        f.write(i+"\t".expandtabs(4))
                    f.write("\n")
            
            shutil.copyfile(path_rai+'/dataset'+Dappend+'/'+model_dir+'Sets.txt',path_rai+'/logs/'+self.timestamp+'/'+model_dir[:15]+'-Sets'+Setappend+'.txt')

        print("------Train Network. "+estimateT(self.epochs_inst+self.epochs_grasp+self.epochs_place)+"------")
        print("Train Instruct for "+str(self.epochs_inst)+" epochs (hidden layers "+str(self.hlayers_inst)+", size "+str(self.size_inst)+"). "+estimateT(self.epochs_inst))
        modelInstructHist=self.modelInstruct.fit(x=dataInput, y={"instructOut": dataInstruct}, epochs=self.epochs_inst,
                                                shuffle=True, verbose=0, validation_split=self.val_split, callbacks=[tbInstruction, printEpoch(),keras.callbacks.LearningRateScheduler(self.step_decay),saveInst, EarlyStopping(self.val_split)])
        

        print("Train Grasp for "+str(self.epochs_grasp)+" epochs (hidden layers "+str(self.hlayers_grasp)+", size "+str(self.size_grasp)+"). "+estimateT(self.epochs_grasp))
        modelGraspHist=self.modelGrasp.fit(x=graspFinalIn,
                                        y={"graspGripperOut": graspFinalOut[:,1,self.listLog[0]], "graspObjectOut": graspFinalOut[:,0,self.listLog[1]]},
                                        epochs=self.epochs_grasp, shuffle=True, verbose=0, validation_split=self.val_split, callbacks=[tbGrasp, printEpoch(),keras.callbacks.LearningRateScheduler(self.step_decay),savegrasp, EarlyStopping(self.val_split)])
        
        
        print("Train Place for "+str(self.epochs_place)+" epochs (hidden layers "+str(self.hlayers_place)+", size "+str(self.size_place)+"). "+estimateT(self.epochs_place))
        modelPlaceHist=self.modelPlace.fit(x=placeFinalIn,
                                        y={"placeGripperOut": placeFinalOut[:,1,self.listLog[0]], "placeObjectOut": placeFinalOut[:,0,self.listLog[1]], "placeTableOut": placeFinalOut[:,2,self.listLog[2]]},
                                        epochs=self.epochs_place, shuffle=True, verbose=0, validation_split=self.val_split, callbacks=[tbPlace, printEpoch(),keras.callbacks.LearningRateScheduler(self.step_decay),saveplace, EarlyStopping(self.val_split)])
        

        if saveToFile:
            self.modelInstruct.save(path_rai+'/logs/'+self.timestamp+'/modelInstruct.h5')
            self.modelGrasp.save(path_rai+'/logs/'+self.timestamp+'/modelGrasp.h5')
            self.modelPlace.save(path_rai+'/logs/'+self.timestamp+'/modelPlace.h5')
            shutil.rmtree(path_rai+'/logs/'+self.timestamp+'/tmp', ignore_errors=True)

        return modelInstructHist, modelGraspHist, modelPlaceHist

class ClassifierChainNN():
    def __init__(self): #no itialization -> either load or build
        pass

    def load_net(self,path_rai, model_dir=''):
        if not model_dir=='':
            model_dir=model_dir+'_full/'
        self.modelInstruct=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelInstruct.h5')
        self.modelGrasp=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelGrasp.h5')
        self.modelPlace=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelPlace.h5')

    def build_net(self,
                inputlength,
                numInstruct,
                numLogicals,
                epochs_inst=300,
                n_layers_inst=2, 
                size_inst=200,
                epochs_grasp=300,
                n_layers_grasp=1,
                size_grasp=200,
                epochs_place=300,
                n_layers_place=1,
                size_place=200,
                timestamp="",
                lr=0.001,
                lr_drop=0.5,
                epoch_drop=100,
                clipnorm=1.,
                val_split=0.2,
                reg=0.001,
                listLog=[]
                ):

        self.inputlength=inputlength
        self.numInstruct=numInstruct
        self.numLogicals=numLogicals

        self.epochs_inst=epochs_inst
        self.hlayers_inst=n_layers_inst
        self.size_inst=size_inst

        self.epochs_inst=epochs_inst
        self.hlayers_inst=n_layers_inst
        self.size_inst=size_inst

        self.epochs_grasp=epochs_grasp
        self.hlayers_grasp=n_layers_grasp
        if self.hlayers_grasp<1:
            self.hlayers_grasp=1
        self.size_grasp=size_grasp

        self.epochs_place=epochs_place
        self.hlayers_place=n_layers_place
        if self.hlayers_place<1:
            self.hlayers_place=1
        self.size_place=size_place

        self.lr=lr
        self.lr_drop=lr_drop
        self.epoch_drop=epoch_drop
        self.val_split=val_split
        self.reg=reg

        self.listLog=listLog
        if self.listLog is []:
            for _ in range(3):
                self.listLog.append(list(range(numLogicals)))

        self.params=[]
        param=["learning rate"]
        param.append("init: "+str(self.lr))
        param.append("drop: "+str(self.lr_drop))
        param.append("epoch_drop: "+str(self.epoch_drop))
        param.append("clipnorm: "+str(clipnorm))
        param.append("val_split: "+str(self.val_split))
        param.append("reg l2: "+str(self.reg))
        self.params.append(param)
        
        if timestamp=="":
            now=datetime.datetime.now()
            self.timestamp=str(now.year)+str(now.month).zfill(2)+str(now.day).zfill(2)+"-"+str(now.hour).zfill(2)+str(now.minute).zfill(2)+str(now.second).zfill(2)+"_full"
        else:
            self.timestamp=timestamp

        self.modelInstruct = self.build_Instruct()
        self.modelGrasp    = self.build_Grasp()
        self.modelPlace    = self.build_Place()

        lossesInstruct={"instructOut": "categorical_crossentropy"}
        lossesGrasp={"graspGripperOut": "categorical_crossentropy", "graspObjectOut": "categorical_crossentropy"}
        lossesPlace={"placeGripperOut": "categorical_crossentropy", "placeObjectOut": "categorical_crossentropy", "placeTableOut": "categorical_crossentropy"}

        weightsInstruct={"instructOut": 1.0}
        weightsGrasp={"graspGripperOut": 0.5, "graspObjectOut": 0.5}
        weightsPlace={"placeGripperOut": 0.33, "placeObjectOut": 0.33, "placeTableOut": 0.33}

        self.modelInstruct.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss=lossesInstruct, loss_weights=weightsInstruct) #decay=0.001/self.epochs_inst
        self.modelGrasp.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss=lossesGrasp, loss_weights=weightsGrasp)#decay=0.001/self.epochs_grasp
        self.modelPlace.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss=lossesPlace, loss_weights=weightsPlace)#decay=0.001/self.epochs_place


    #------------------build subnetwork---------------

    def build_Instruct(self):
        # One Hot : Multi label classification
        param=["Instruct"]
        param.append("hlayers_inst: "+str(self.hlayers_inst))
        param.append("size_inst: "+str(self.size_inst))
        param.append("epochs_inst: "+str(self.epochs_inst))
        self.params.append(param)

        inputs = keras.Input(shape=(self.inputlength,))
        InstructNet = build_Sub(inputs, self.numInstruct, "instructScope", "instructOut", n_layers=self.hlayers_inst, size=self.size_inst)
        
        modelInstuct = keras.models.Model(inputs=inputs, outputs=[InstructNet], name="instructionNet")
        return modelInstuct

    def build_Grasp(self):
        # Multi Hot: Multi class classification through 2 steps
        param=["Grasp"]
        param.append("hlayers_grasp: "+str(self.hlayers_grasp))
        param.append("size_grasp: "+str(self.size_grasp))
        param.append("epochs_grasp: "+str(self.epochs_grasp))
        self.params.append(param)
        
        #inputs = keras.Input(shape=(2,self.inputlength+2*self.numLogicals))
        inputs = keras.Input(shape=(1,self.inputlength))
        objout, ObjNet = build_LSTM(inputs, len(self.listLog[1]), "graspObjectScope", "graspObjectOut", n_layers=self.hlayers_grasp, size=self.size_grasp, reg=self.reg)

        GripperInputs = keras.layers.concatenate([inputs, objout])
        _, GripperNet = build_LSTM(GripperInputs, len(self.listLog[0]), "graspGripperScope", "graspGripperOut", n_layers=self.hlayers_grasp, size=self.size_grasp, reg=self.reg)


        modelGrasp = keras.models.Model(inputs=inputs, outputs=[GripperNet, ObjNet], name="graspNet")

        return modelGrasp

    def build_Place(self):
        # Multi Hot: Multi class classification through 3 steps
        param=["Place"]
        param.append("hlayers_place: "+str(self.hlayers_place))
        param.append("size_place: "+str(self.size_place))
        param.append("epochs_place: "+str(self.epochs_place))
        self.params.append(param)

        #inputs = keras.Input(shape=(3,self.inputlength+3*self.numLogicals))
        inputs = keras.Input(shape=(1,self.inputlength))
        objout, ObjNet = build_LSTM(inputs, len(self.listLog[1]), "placeObjectScope", "placeObjectOut", n_layers=self.hlayers_place, size=self.size_place, reg=self.reg)

        GripperInputs = keras.layers.concatenate([inputs, objout])
        grout, GripperNet = build_LSTM(GripperInputs, len(self.listLog[0]), "placeGripperScope", "placeGripperOut", n_layers=self.hlayers_place, size=self.size_place, reg=self.reg)

        TableInputs = keras.layers.concatenate([GripperInputs, grout])
        _, TableNet = build_LSTM(TableInputs, len(self.listLog[2]), "placeTableScope", "placeTableOut", n_layers=self.hlayers_place, size=self.size_place, reg=self.reg)


        modelPlace = keras.models.Model(inputs=inputs, outputs=[GripperNet, ObjNet, TableNet], name="placeNet")
        return modelPlace

    #------------------Train and save Model---------------
    #----TODO train, adapt input etc

    def reshapeInput(self, path_rai, model_dir):

        dataInput=np.load(path_rai+'/dataset/'+model_dir+'Input.npy')
        dataInstruct=np.load(path_rai+'/dataset/'+model_dir+'Instruction.npy')
        dataLogicals=np.load(path_rai+'/dataset/'+model_dir+'Logicals.npy')

        idx=list(range(dataInput.shape[0]))
        random.shuffle(idx)

        dataInput=dataInput[idx,:]
        dataInstruct=dataInstruct[idx,:]
        dataLogicals=dataLogicals[idx,:]

        #-------------------------------------------
        #----grasp gripper object

        graspIn=dataInput[dataInstruct[:,0]==1,:]
        graspAct=dataLogicals[dataInstruct[:,0]==1,:].astype('int')

        # object
        #graspIn1=np.concatenate((graspIn, np.zeros((graspAct.shape[0], 2*self.numLogicals))), axis=1)
        #graspIn1=np.concatenate((graspIn, np.zeros((graspAct.shape[0], 1)), np.ones((graspAct.shape[0], 1))), axis=1)
        graspOut1=graspAct[:,self.numLogicals:2*self.numLogicals] #object
        
        # gripper
        #graspIn2=np.concatenate((graspIn, np.zeros((graspAct.shape[0], self.numLogicals)), graspAct[:,self.numLogicals:2*self.numLogicals]), axis=1)
        #graspIn2=np.concatenate((graspIn, np.ones((graspAct.shape[0], 1)), np.zeros((graspAct.shape[0], 1))), axis=1)
        graspOut2=graspAct[:,:self.numLogicals] #gripper

        #graspFinalIn=np.zeros((2*graspIn.shape[0], graspIn.shape[1]+2*self.numLogicals))
        #graspFinalIn=np.zeros((2*graspIn.shape[0], graspIn.shape[1]+2))

        graspFinalIn=graspIn.reshape(-1,1, graspIn.shape[1])

        """print(graspIn1.shape)
        graspFinalIn[0::2,:]=graspIn1
        graspFinalIn[1::2,:]=graspIn2 """

        graspFinalOut=np.zeros((2*graspIn.shape[0], self.numLogicals))
        graspFinalOut[0::2,:]=graspOut1
        graspFinalOut[1::2,:]=graspOut2

        #-------------------------------------------
        #----place gripper object table

        placeIn=dataInput[dataInstruct[:,1]==1,:]
        placeAct=dataLogicals[dataInstruct[:,1]==1,:].astype('int')

        # object
        #placeIn1=np.concatenate((placeIn, np.zeros(placeAct.shape)), axis=1)
        #placeIn1=np.concatenate((placeIn, np.zeros((placeAct.shape[0], 1)), np.ones((placeAct.shape[0], 1)), np.zeros((placeAct.shape[0], 1))), axis=1)
        placeOut1=placeAct[:,self.numLogicals:2*self.numLogicals] #object

        # gripper
        #placeIn2=np.concatenate((placeIn, np.zeros((placeAct.shape[0], self.numLogicals)), placeAct[:,self.numLogicals:2*self.numLogicals], np.zeros((placeAct.shape[0], self.numLogicals))), axis=1)
        #placeIn2=np.concatenate((placeIn, np.ones((placeAct.shape[0], 1)), np.zeros((placeAct.shape[0], 1)), np.zeros((placeAct.shape[0], 1))), axis=1)
        placeOut2=placeAct[:,:self.numLogicals] #gripper

        # table
        #placeIn3=np.concatenate((placeIn, placeAct[:,0:2*self.numLogicals], np.zeros((placeAct.shape[0], self.numLogicals))), axis=1)
        #placeIn3=np.concatenate((placeIn, np.zeros((placeAct.shape[0], 1)), np.zeros((placeAct.shape[0], 1)), np.ones((placeAct.shape[0], 1))), axis=1)
        placeOut3=placeAct[:,2*self.numLogicals:] #table

        #placeFinalIn=np.zeros((3*placeIn.shape[0], placeIn.shape[1]+3*self.numLogicals))
        #placeFinalIn=np.zeros((3*placeIn.shape[0], placeIn.shape[1]+3))
        placeFinalIn=placeIn.reshape(-1,1, placeIn.shape[1])


        """print(placeIn1.shape)
        placeFinalIn[0::3,:]=placeIn1
        placeFinalIn[1::3,:]=placeIn2
        placeFinalIn[2::3,:]=placeIn3 """

        placeFinalOut=np.zeros((3*placeIn.shape[0], self.numLogicals))
        placeFinalOut[0::3,:]=placeOut1
        placeFinalOut[1::3,:]=placeOut2
        placeFinalOut[2::3,:]=placeOut3

        #return dataInput, dataInstruct, graspFinalIn.reshape(-1,2,graspIn.shape[1]+2*self.numLogicals), graspFinalOut.reshape(-1,2,self.numLogicals), placeFinalIn.reshape(-1,3,placeIn.shape[1]+3*self.numLogicals), placeFinalOut.reshape(-1,3,self.numLogicals)
        return dataInput, dataInstruct, graspFinalIn, graspFinalOut.reshape((-1,2,self.numLogicals)), placeFinalIn, placeFinalOut.reshape((-1,3,self.numLogicals))
        

    def step_decay(self,epoch):
        initial_lrate = self.lr
        drop = self.lr_drop
        epochs_drop = self.epoch_drop
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate


    def train(self, path_rai, model_dir, saveToFile=True):
        if not model_dir=='':
            model_dir=model_dir+'_full/'

        dataInput, dataInstruct, graspFinalIn, graspFinalOut, placeFinalIn, placeFinalOut=self.reshapeInput(path_rai, model_dir)
        #print(np.any(np.isnan(dataInput)))
        #print(np.any(np.isnan(dataInstruct)))
        #print(np.any(np.isnan(graspFinalIn)))
        #print(np.any(np.isnan(graspFinalOut)))
        #print(np.any(np.isnan(placeFinalIn)))
        #print(np.any(np.isnan(placeFinalOut)))

        tf.keras.backend.get_session().run(tf.global_variables_initializer())

        tbInstruction = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/Instruction')
        tbGrasp = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/Grasp')
        tbPlace = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/Place')

        if not(os.path.exists(path_rai+'/logs/'+self.timestamp+'/tmp')):
                os.makedirs(path_rai+'/logs/'+self.timestamp+'/tmp')

        saveInst=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelInstruct.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        savegrasp=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelGrasp.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        saveplace=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelPlace.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)

        if saveToFile:
            if not(os.path.exists(path_rai+'/logs/'+self.timestamp)):
                os.makedirs(path_rai+'/logs/'+self.timestamp)

            with open(path_rai+'/logs/'+self.timestamp+'/params.txt', 'w') as f:
                for item in self.params:
                    for i in item:
                        f.write(i+"\t".expandtabs(4))
                    f.write("\n")
            
            shutil.copyfile(path_rai+'/dataset/'+model_dir+'Sets.txt',path_rai+'/logs/'+self.timestamp+'/'+model_dir[:15]+'-Sets.txt')

        print("------Train Network. "+estimateT(self.epochs_inst)+" + "+estimateT_lstm(self.epochs_grasp+self.epochs_place)+"------")

        print("Train Grasp for "+str(self.epochs_grasp)+" epochs (hidden layers "+str(self.hlayers_grasp)+", size "+str(self.size_grasp)+"). "+estimateT_lstm(self.epochs_grasp))
        modelGraspHist=self.modelGrasp.fit(x=graspFinalIn, y={"graspGripperOut":graspFinalOut[:,1,self.listLog[0]], "graspObjectOut":graspFinalOut[:,0,self.listLog[1]]}, batch_size=32,
                                        epochs=self.epochs_grasp, shuffle=True, verbose=0, validation_split=self.val_split, callbacks=[tbGrasp, printEpoch(),savegrasp, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])
        
        
        print("Train Place for "+str(self.epochs_place)+" epochs (hidden layers "+str(self.hlayers_place)+", size "+str(self.size_place)+"). "+estimateT_lstm(self.epochs_place))
        modelPlaceHist=self.modelPlace.fit(x=placeFinalIn, y={"placeGripperOut": placeFinalOut[:,1,self.listLog[0]], "placeObjectOut": placeFinalOut[:,0,self.listLog[1]], "placeTableOut": placeFinalOut[:,2,self.listLog[2]]},batch_size=32,
                                        epochs=self.epochs_place, shuffle=True, verbose=0, validation_split=self.val_split, callbacks=[tbPlace, printEpoch(),saveplace, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])
        
        print("Train Instruct for "+str(self.epochs_inst)+" epochs (hidden layers "+str(self.hlayers_inst)+", size "+str(self.size_inst)+"). "+estimateT(self.epochs_inst))
        modelInstructHist=self.modelInstruct.fit(x=dataInput, y={"instructOut": dataInstruct}, batch_size=32, epochs=self.epochs_inst,
                                                shuffle=True, verbose=0, validation_split=self.val_split, callbacks=[tbInstruction, printEpoch(),saveInst, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])
        

        if saveToFile:
            self.modelInstruct.save(path_rai+'/logs/'+self.timestamp+'/modelInstruct.h5')
            self.modelGrasp.save(path_rai+'/logs/'+self.timestamp+'/modelGrasp.h5')
            self.modelPlace.save(path_rai+'/logs/'+self.timestamp+'/modelPlace.h5')
            shutil.rmtree(path_rai+'/logs/'+self.timestamp+'/tmp', ignore_errors=True)

        return modelInstructHist, modelGraspHist, modelPlaceHist

#----------------------------------------------------------------------------

class ClassifierChainNNSplit():
    def __init__(self): #no itialization -> either load or build
        pass

    def load_net(self,path_rai, model_dir=''):
        if not model_dir=='':
            model_dir=model_dir+'_chain/'
        self.modelInstruct=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelInstruct.h5')
        self.modelGraspObj=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelGraspObj.h5')
        self.modelGraspGrp=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelGraspGrp.h5')

        self.modelPlaceObj=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelPlaceObj.h5')
        self.modelPlaceGrp=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelPlaceGrp.h5')
        self.modelPlaceTab=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelPlaceTab.h5')


    def build_net(self,
                inputlength,
                numInstruct,
                numLogicals,
                epochs_inst=300,
                n_layers_inst=2, 
                size_inst=200,
                epochs_grasp=300,
                n_layers_grasp=1,
                size_grasp=200,
                epochs_place=300,
                n_layers_place=1,
                size_place=200,
                timestamp="",
                lr=0.001,
                lr_drop=0.5,
                epoch_drop=100,
                clipnorm=1.,
                val_split=0.2,
                reg=0.001,
                listLog=[]
                ):

        self.inputlength=inputlength
        self.numInstruct=numInstruct
        self.numLogicals=numLogicals

        self.epochs_inst=epochs_inst
        self.hlayers_inst=n_layers_inst
        self.size_inst=size_inst

        self.epochs_inst=epochs_inst
        self.hlayers_inst=n_layers_inst
        self.size_inst=size_inst

        self.epochs_grasp=epochs_grasp
        self.hlayers_grasp=n_layers_grasp
        if self.hlayers_grasp<1:
            self.hlayers_grasp=1
        self.size_grasp=size_grasp

        self.epochs_place=epochs_place
        self.hlayers_place=n_layers_place
        if self.hlayers_place<1:
            self.hlayers_place=1
        self.size_place=size_place

        self.lr=lr
        self.lr_drop=lr_drop
        self.epoch_drop=epoch_drop
        self.val_split=val_split
        self.reg=reg

        self.listLog=listLog
        if self.listLog is []:
            for _ in range(3):
                self.listLog.append(list(range(numLogicals)))

        print(self.listLog)

        self.params=[]
        param=["learning rate"]
        param.append("init: "+str(self.lr))
        param.append("drop: "+str(self.lr_drop))
        param.append("epoch_drop: "+str(self.epoch_drop))
        param.append("clipnorm: "+str(clipnorm))
        param.append("val_split: "+str(self.val_split))
        param.append("reg l2: "+str(self.reg))
        self.params.append(param)
        
        if timestamp=="":
            now=datetime.datetime.now()
            self.timestamp=str(now.year)+str(now.month).zfill(2)+str(now.day).zfill(2)+"-"+str(now.hour).zfill(2)+str(now.minute).zfill(2)+str(now.second).zfill(2)+"_chain"
        else:
            self.timestamp=timestamp

        self.modelInstruct = self.build_Instruct()
        self.modelGraspObj, self.modelGraspGrp = self.build_Grasp()
        self.modelPlaceObj, self.modelPlaceGrp, self.modelPlaceTab = self.build_Place()

        #lossesInstruct={"instructOut": "categorical_crossentropy"}
        #lossesGrasp={"graspGripperOut": "categorical_crossentropy", "graspObjectOut": "categorical_crossentropy"}
        #lossesPlace={"placeGripperOut": "categorical_crossentropy", "placeObjectOut": "categorical_crossentropy", "placeTableOut": "categorical_crossentropy"}

        #weightsInstruct={"instructOut": 1.0}
        #weightsGrasp={"graspGripperOut": 0.5, "graspObjectOut": 0.5}
        #weightsPlace={"placeGripperOut": 0.33, "placeObjectOut": 0.33, "placeTableOut": 0.33}

        self.modelInstruct.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy") #decay=0.001/self.epochs_inst
        self.modelGraspObj.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_grasp
        self.modelGraspGrp.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_grasp

        self.modelPlaceObj.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_place
        self.modelPlaceGrp.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_place
        self.modelPlaceTab.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_grasp

    #------------------build subnetwork---------------

    def build_Instruct(self):
        # One Hot : Multi label classification
        param=["Instruct"]
        param.append("hlayers_inst: "+str(self.hlayers_inst))
        param.append("size_inst: "+str(self.size_inst))
        param.append("epochs_inst: "+str(self.epochs_inst))
        self.params.append(param)

        inputs = keras.Input(shape=(self.inputlength,))
        InstructNet = build_Sub(inputs, self.numInstruct, "instructScope", "instructOut", n_layers=self.hlayers_inst, size=self.size_inst)
        
        modelInstuct = keras.models.Model(inputs=inputs, outputs=[InstructNet], name="instructionNet")
        return modelInstuct

    def build_Grasp(self):
        # Multi Hot: Multi class classification through 2 steps
        param=["Grasp"]
        param.append("hlayers_grasp: "+str(self.hlayers_grasp))
        param.append("size_grasp: "+str(self.size_grasp))
        param.append("epochs_grasp: "+str(self.epochs_grasp))
        self.params.append(param)
        
        #inputs = keras.Input(shape=(2,self.inputlength+2*self.numLogicals))
        inputs1 = keras.Input(shape=(1,self.inputlength))
        _, ObjNet = build_LSTM(inputs1, len(self.listLog[1]), "graspObjectScope", "graspObjectOut", n_layers=self.hlayers_grasp, size=self.size_grasp, reg=self.reg)

        inputs2 = keras.Input(shape=(1,self.inputlength+self.numLogicals))
        _, GrpNet = build_LSTM(inputs2, len(self.listLog[0]), "graspGripperScope", "graspGripperOut", n_layers=self.hlayers_grasp, size=self.size_grasp, reg=self.reg)


        modelGraspObj = keras.models.Model(inputs=inputs1, outputs=ObjNet, name="graspObjNet")
        modelGraspGrp = keras.models.Model(inputs=inputs2, outputs=GrpNet, name="graspGrpNet")

        return modelGraspObj, modelGraspGrp

    def build_Place(self):
        # Multi Hot: Multi class classification through 3 steps
        param=["Place"]
        param.append("hlayers_place: "+str(self.hlayers_place))
        param.append("size_place: "+str(self.size_place))
        param.append("epochs_place: "+str(self.epochs_place))
        self.params.append(param)

        #inputs = keras.Input(shape=(3,self.inputlength+3*self.numLogicals))
        inputs1 = keras.Input(shape=(1,self.inputlength))
        _, ObjNet = build_LSTM(inputs1, len(self.listLog[1]), "placeObjectScope", "placeObjectOut", n_layers=self.hlayers_place, size=self.size_place, reg=self.reg)

        inputs2 = keras.Input(shape=(1,self.inputlength+self.numLogicals))
        _, GrpNet = build_LSTM(inputs2, len(self.listLog[0]), "placeGripperScope", "placeGripperOut", n_layers=self.hlayers_place, size=self.size_place, reg=self.reg)

        inputs3 = keras.Input(shape=(1,self.inputlength+2*self.numLogicals))
        _, TabNet = build_LSTM(inputs3, len(self.listLog[2]), "placeTableScope", "placeTableOut", n_layers=self.hlayers_place, size=self.size_place, reg=self.reg)


        modelPlaceObj = keras.models.Model(inputs=inputs1, outputs=ObjNet, name="placeObjeNet")
        modelPlaceGrp = keras.models.Model(inputs=inputs2, outputs=GrpNet, name="placeGrpNet")
        modelPlaceTab = keras.models.Model(inputs=inputs3, outputs=TabNet, name="placeTabNet")

        return modelPlaceObj, modelPlaceGrp, modelPlaceTab

    #------------------Train and save Model---------------
    #----TODO train, adapt input etc

    def reshapeInput(self, path_rai, model_dir):

        dataInput=np.load(path_rai+'/dataset/'+model_dir+'Input.npy')
        dataInstruct=np.load(path_rai+'/dataset/'+model_dir+'Instruction.npy')
        dataLogicals=np.load(path_rai+'/dataset/'+model_dir+'Logicals.npy')

        idx=list(range(dataInput.shape[0]))
        random.shuffle(idx)

        dataInput=dataInput[idx,:]
        dataInstruct=dataInstruct[idx,:]
        dataLogicals=dataLogicals[idx,:]

        #-------------------------------------------
        #----grasp gripper object

        graspIn=dataInput[dataInstruct[:,0]==1,:]
        graspAct=dataLogicals[dataInstruct[:,0]==1,:].astype('int')

        # object
        #graspIn1=np.concatenate((graspIn, np.zeros((graspAct.shape[0], 2*self.numLogicals))), axis=1)
        #graspIn1=np.concatenate((graspIn, np.zeros((graspAct.shape[0], 1)), np.ones((graspAct.shape[0], 1))), axis=1)
        graspOut1=graspAct[:,self.numLogicals:2*self.numLogicals] #object
        
        # gripper
        #graspIn2=np.concatenate((graspIn, np.zeros((graspAct.shape[0], self.numLogicals)), graspAct[:,self.numLogicals:2*self.numLogicals]), axis=1)
        #graspIn2=np.concatenate((graspIn, np.ones((graspAct.shape[0], 1)), np.zeros((graspAct.shape[0], 1))), axis=1)
        graspOut2=graspAct[:,:self.numLogicals] #gripper

        #graspFinalIn=np.zeros((2*graspIn.shape[0], graspIn.shape[1]+2*self.numLogicals))
        #graspFinalIn=np.zeros((2*graspIn.shape[0], graspIn.shape[1]+2))

        graspFinalIn=graspIn.reshape(-1,1, graspIn.shape[1])

        """print(graspIn1.shape)
        graspFinalIn[0::2,:]=graspIn1
        graspFinalIn[1::2,:]=graspIn2 """

        graspFinalOut=np.zeros((2*graspIn.shape[0], self.numLogicals))
        graspFinalOut[0::2,:]=graspOut1
        graspFinalOut[1::2,:]=graspOut2

        #-------------------------------------------
        #----place gripper object table

        placeIn=dataInput[dataInstruct[:,1]==1,:]
        placeAct=dataLogicals[dataInstruct[:,1]==1,:].astype('int')

        # object
        #placeIn1=np.concatenate((placeIn, np.zeros(placeAct.shape)), axis=1)
        #placeIn1=np.concatenate((placeIn, np.zeros((placeAct.shape[0], 1)), np.ones((placeAct.shape[0], 1)), np.zeros((placeAct.shape[0], 1))), axis=1)
        placeOut1=placeAct[:,self.numLogicals:2*self.numLogicals] #object

        # gripper
        #placeIn2=np.concatenate((placeIn, np.zeros((placeAct.shape[0], self.numLogicals)), placeAct[:,self.numLogicals:2*self.numLogicals], np.zeros((placeAct.shape[0], self.numLogicals))), axis=1)
        #placeIn2=np.concatenate((placeIn, np.ones((placeAct.shape[0], 1)), np.zeros((placeAct.shape[0], 1)), np.zeros((placeAct.shape[0], 1))), axis=1)
        placeOut2=placeAct[:,:self.numLogicals] #gripper

        # table
        #placeIn3=np.concatenate((placeIn, placeAct[:,0:2*self.numLogicals], np.zeros((placeAct.shape[0], self.numLogicals))), axis=1)
        #placeIn3=np.concatenate((placeIn, np.zeros((placeAct.shape[0], 1)), np.zeros((placeAct.shape[0], 1)), np.ones((placeAct.shape[0], 1))), axis=1)
        placeOut3=placeAct[:,2*self.numLogicals:] #table

        #placeFinalIn=np.zeros((3*placeIn.shape[0], placeIn.shape[1]+3*self.numLogicals))
        #placeFinalIn=np.zeros((3*placeIn.shape[0], placeIn.shape[1]+3))
        placeFinalIn=placeIn.reshape(-1,1, placeIn.shape[1])


        """print(placeIn1.shape)
        placeFinalIn[0::3,:]=placeIn1
        placeFinalIn[1::3,:]=placeIn2
        placeFinalIn[2::3,:]=placeIn3 """

        placeFinalOut=np.zeros((3*placeIn.shape[0], self.numLogicals))
        placeFinalOut[0::3,:]=placeOut1
        placeFinalOut[1::3,:]=placeOut2
        placeFinalOut[2::3,:]=placeOut3

        #return dataInput, dataInstruct, graspFinalIn.reshape(-1,2,graspIn.shape[1]+2*self.numLogicals), graspFinalOut.reshape(-1,2,self.numLogicals), placeFinalIn.reshape(-1,3,placeIn.shape[1]+3*self.numLogicals), placeFinalOut.reshape(-1,3,self.numLogicals)
        return dataInput, dataInstruct, graspFinalIn, graspFinalOut.reshape((-1,2,self.numLogicals)), placeFinalIn, placeFinalOut.reshape((-1,3,self.numLogicals))
        

    def step_decay(self,epoch):
        initial_lrate = self.lr
        drop = self.lr_drop
        epochs_drop = self.epoch_drop
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate


    def train(self, path_rai, model_dir, saveToFile=True):
        if not model_dir=='':
            model_dir=model_dir+'_full/'

        dataInput, dataInstruct, graspFinalIn, graspFinalOut, placeFinalIn, placeFinalOut=self.reshapeInput(path_rai, model_dir)
        #print(np.any(np.isnan(dataInput)))
        #print(np.any(np.isnan(dataInstruct)))
        #print(np.any(np.isnan(graspFinalIn)))
        #print(np.any(np.isnan(graspFinalOut)))
        #print(np.any(np.isnan(placeFinalIn)))
        #print(np.any(np.isnan(placeFinalOut)))

        tf.keras.backend.get_session().run(tf.global_variables_initializer())

        tbInstruction = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/Instruction')
        tbGraspObj = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/GraspObj')
        tbGraspGrp = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/GraspGrp')

        tbPlaceObj = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/PlaceObj')
        tbPlaceGrp = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/PlaceGrp')
        tbPlaceTab = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/PlaceTab')

        if not(os.path.exists(path_rai+'/logs/'+self.timestamp+'/tmp')):
                os.makedirs(path_rai+'/logs/'+self.timestamp+'/tmp')

        saveInst=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelInstruct.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        savegraspObj=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelGraspObj.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        savegraspGrp=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelGraspGrp.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)

        saveplaceObj=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelPlaceObj.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        saveplaceGrp=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelPlaceGrp.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        saveplaceTab=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelPlaceTab.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)

        if saveToFile:
            if not(os.path.exists(path_rai+'/logs/'+self.timestamp)):
                os.makedirs(path_rai+'/logs/'+self.timestamp)

            with open(path_rai+'/logs/'+self.timestamp+'/params.txt', 'w') as f:
                for item in self.params:
                    for i in item:
                        f.write(i+"\t".expandtabs(4))
                    f.write("\n")
            
            shutil.copyfile(path_rai+'/dataset/'+model_dir+'Sets.txt',path_rai+'/logs/'+self.timestamp+'/'+model_dir[:15]+'-Sets.txt')


        print("------Train Network. "+estimateT(self.epochs_inst)+" + "+estimateT_lstm(self.epochs_grasp+self.epochs_place, num=5)+"------")

        print("Train Grasp for "+str(self.epochs_grasp)+" epochs (hidden layers "+str(self.hlayers_grasp)+", size "+str(self.size_grasp)+"). "+estimateT_lstm(self.epochs_grasp, num=2))
        #graspShape0=graspFinalOut.shape[0]
        print("GraspObj")
        modelGraspObjHist=self.modelGraspObj.fit(x=graspFinalIn, y={"graspObjectOut":graspFinalOut[:,0,self.listLog[1]]}, batch_size=32,
                                        epochs=self.epochs_grasp, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbGraspObj,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),savegraspObj, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])
        print("GraspGrp")
        modelGraspGrpHist=self.modelGraspGrp.fit(x=np.concatenate((graspFinalIn, graspFinalOut[:,0:1,:]), axis=2), y={"graspGripperOut":graspFinalOut[:,1,self.listLog[0]]}, batch_size=32,
                                        epochs=self.epochs_grasp, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbGraspGrp,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),savegraspGrp, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])

        print("Train Place for "+str(self.epochs_place)+" epochs (hidden layers "+str(self.hlayers_place)+", size "+str(self.size_place)+"). "+estimateT_lstm(self.epochs_place, num=3))      
        #placeShape0=placeFinalOut.shape[0]
        print("PlaceObj")
        modelPlaceObjHist=self.modelPlaceObj.fit(x=placeFinalIn, y={"placeObjectOut": placeFinalOut[:,0,self.listLog[1]]},batch_size=32,
                                        epochs=self.epochs_place, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbPlaceObj,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),saveplaceObj, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])
        print("PlaceGrp")
        modelPlaceGrpHist=self.modelPlaceGrp.fit(x=np.concatenate((placeFinalIn, placeFinalOut[:,0:1,:]), axis=2), y={"placeGripperOut": placeFinalOut[:,1,self.listLog[0]]},batch_size=32,
                                        epochs=self.epochs_place, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbPlaceGrp,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),saveplaceGrp, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])
        print("PlaceTab")
        modelPlaceTabHist=self.modelPlaceTab.fit(x=np.concatenate((placeFinalIn, placeFinalOut[:,0:1,:], placeFinalOut[:,1:2,:]), axis=2), y={"placeTableOut": placeFinalOut[:,2,self.listLog[2]]},batch_size=32,
                                        epochs=self.epochs_place, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbPlaceTab,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),saveplaceTab, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])


        print("Train Instruct for "+str(self.epochs_inst)+" epochs (hidden layers "+str(self.hlayers_inst)+", size "+str(self.size_inst)+"). "+estimateT(self.epochs_inst))
        modelInstructHist=self.modelInstruct.fit(x=dataInput, y={"instructOut": dataInstruct}, batch_size=32, epochs=self.epochs_inst,
                                                shuffle=True, verbose=0, validation_split=self.val_split,
                                                callbacks=[tbInstruction,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),saveInst, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])
        

        if saveToFile:
            self.modelInstruct.save(path_rai+'/logs/'+self.timestamp+'/modelInstruct.h5')
            self.modelGraspObj.save(path_rai+'/logs/'+self.timestamp+'/modelGraspObj.h5')
            self.modelGraspGrp.save(path_rai+'/logs/'+self.timestamp+'/modelGraspGrp.h5')
            self.modelPlaceObj.save(path_rai+'/logs/'+self.timestamp+'/modelPlaceObj.h5')
            self.modelPlaceGrp.save(path_rai+'/logs/'+self.timestamp+'/modelPlaceGrp.h5')
            self.modelPlaceTab.save(path_rai+'/logs/'+self.timestamp+'/modelPlaceTab.h5')
            shutil.rmtree(path_rai+'/logs/'+self.timestamp+'/tmp', ignore_errors=True)

        return modelInstructHist, modelGraspObjHist, modelGraspGrpHist, modelPlaceObjHist, modelPlaceGrpHist, modelPlaceTabHist


class ClassifierChainFFNNSplit():
    def __init__(self): #no itialization -> either load or build
        pass

    def load_net(self,path_rai, model_dir=''):
        if not model_dir=='':
            model_dir=model_dir+'_FFchain/'
        self.modelInstruct=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelInstruct.h5')
        self.modelGraspObj=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelGraspObj.h5')
        self.modelGraspGrp=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelGraspGrp.h5')

        self.modelPlaceObj=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelPlaceObj.h5')
        self.modelPlaceGrp=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelPlaceGrp.h5')
        self.modelPlaceTab=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelPlaceTab.h5')


    def build_net(self,
                inputlength,
                numInstruct,
                numLogicals,
                epochs_inst=300,
                n_layers_inst=2, 
                size_inst=200,
                epochs_grasp=300,
                n_layers_grasp=1,
                size_grasp=200,
                epochs_place=300,
                n_layers_place=1,
                size_place=200,
                timestamp="",
                lr=0.001,
                lr_drop=0.5,
                epoch_drop=100,
                clipnorm=1.,
                val_split=0.2,
                reg0=0.001,
                listLog=[],
                mode=1
                ):

        self.inputlength=inputlength
        self.numInstruct=numInstruct
        self.numLogicals=numLogicals

        self.epochs_inst=epochs_inst
        self.hlayers_inst=n_layers_inst
        self.size_inst=size_inst

        self.epochs_inst=epochs_inst
        self.hlayers_inst=n_layers_inst
        self.size_inst=size_inst

        self.epochs_grasp=epochs_grasp
        self.hlayers_grasp=n_layers_grasp
        if self.hlayers_grasp<1:
            self.hlayers_grasp=1
        self.size_grasp=size_grasp

        self.epochs_place=epochs_place
        self.hlayers_place=n_layers_place
        if self.hlayers_place<1:
            self.hlayers_place=1
        self.size_place=size_place

        self.lr=lr
        self.lr_drop=lr_drop
        self.epoch_drop=epoch_drop
        self.val_split=val_split
        self.reg0=reg0
        self.mode=mode

        self.listLog=listLog
        if self.listLog is []:
            for _ in range(3):
                self.listLog.append(list(range(numLogicals)))

        print(self.listLog)

        self.params=[]
        param=["learning rate"]
        param.append("init: "+str(self.lr))
        param.append("drop: "+str(self.lr_drop))
        param.append("epoch_drop: "+str(self.epoch_drop))
        param.append("clipnorm: "+str(clipnorm))
        param.append("val_split: "+str(self.val_split))
        param.append("reg_l2: "+str(reg0))
        param.append("dataset_mode: "+str(self.mode))
        self.params.append(param)
        
        if timestamp=="":
            now=datetime.datetime.now()
            self.timestamp=str(now.year)+str(now.month).zfill(2)+str(now.day).zfill(2)+"-"+str(now.hour).zfill(2)+str(now.minute).zfill(2)+str(now.second).zfill(2)+"_FFchain"
        else:
            self.timestamp=timestamp

        self.modelInstruct = self.build_Instruct()
        self.modelGraspObj, self.modelGraspGrp = self.build_Grasp()
        self.modelPlaceObj, self.modelPlaceGrp, self.modelPlaceTab = self.build_Place()

        #lossesInstruct={"instructOut": "categorical_crossentropy"}
        #lossesGrasp={"graspGripperOut": "categorical_crossentropy", "graspObjectOut": "categorical_crossentropy"}
        #lossesPlace={"placeGripperOut": "categorical_crossentropy", "placeObjectOut": "categorical_crossentropy", "placeTableOut": "categorical_crossentropy"}

        #weightsInstruct={"instructOut": 1.0}
        #weightsGrasp={"graspGripperOut": 0.5, "graspObjectOut": 0.5}
        #weightsPlace={"placeGripperOut": 0.33, "placeObjectOut": 0.33, "placeTableOut": 0.33}

        self.modelInstruct.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy") #decay=0.001/self.epochs_inst
        self.modelGraspObj.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_grasp
        self.modelGraspGrp.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_grasp

        self.modelPlaceObj.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_place
        self.modelPlaceGrp.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_place
        self.modelPlaceTab.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_grasp

    #------------------build subnetwork---------------

    def build_Instruct(self):
        # One Hot : Multi label classification
        param=["Instruct"]
        param.append("hlayers_inst: "+str(self.hlayers_inst))
        param.append("size_inst: "+str(self.size_inst))
        param.append("epochs_inst: "+str(self.epochs_inst))
        self.params.append(param)

        inputs = keras.Input(shape=(self.inputlength,))
        InstructNet = build_Sub(inputs, self.numInstruct, "instructScope", "instructOut", n_layers=self.hlayers_inst, size=self.size_inst)
        
        modelInstuct = keras.models.Model(inputs=inputs, outputs=[InstructNet], name="instructionNet")
        return modelInstuct

    def build_Grasp(self):
        # Multi Hot: Multi class classification through 2 steps
        param=["Grasp"]
        param.append("hlayers_grasp: "+str(self.hlayers_grasp))
        param.append("size_grasp: "+str(self.size_grasp))
        param.append("epochs_grasp: "+str(self.epochs_grasp))
        self.params.append(param)
        
        #inputs = keras.Input(shape=(2,self.inputlength+2*self.numLogicals))
        inputs1 = keras.Input(shape=(self.inputlength,))
        #_, ObjNet = build_LSTM(inputs1, len(self.listLog[1]), "graspObjectScope", "graspObjectOut", n_layers=self.hlayers_grasp, size=self.size_grasp, reg=self.reg)
        ObjNet = build_Sub(inputs1, len(self.listLog[1]), "graspObjectScope", "graspObjectOut", n_layers=self.hlayers_grasp, size=self.size_grasp, reg=self.reg0)

        inputs2 = keras.Input(shape=(self.inputlength+len(self.listLog[1]),))
        #_, GrpNet = build_LSTM(inputs2, len(self.listLog[0]), "graspGripperScope", "graspGripperOut", n_layers=self.hlayers_grasp, size=self.size_grasp, reg=self.reg)
        GrpNet = build_Sub(inputs2, len(self.listLog[0]), "graspGripperScope", "graspGripperOut", n_layers=self.hlayers_grasp, size=self.size_grasp, reg=self.reg0)


        modelGraspObj = keras.models.Model(inputs=inputs1, outputs=ObjNet, name="graspObjNet")
        modelGraspGrp = keras.models.Model(inputs=inputs2, outputs=GrpNet, name="graspGrpNet")

        return modelGraspObj, modelGraspGrp

    def build_Place(self):
        # Multi Hot: Multi class classification through 3 steps
        param=["Place"]
        param.append("hlayers_place: "+str(self.hlayers_place))
        param.append("size_place: "+str(self.size_place))
        param.append("epochs_place: "+str(self.epochs_place))
        self.params.append(param)

        #inputs = keras.Input(shape=(3,self.inputlength+3*self.numLogicals))
        inputs1 = keras.Input(shape=(self.inputlength,))
        #_, ObjNet = build_LSTM(inputs1, len(self.listLog[1]), "placeObjectScope", "placeObjectOut", n_layers=self.hlayers_place, size=self.size_place, reg=self.reg)
        ObjNet = build_Sub(inputs1, len(self.listLog[1]), "placeObjectScope", "placeObjectOut", n_layers=self.hlayers_place, size=self.size_place, reg=self.reg0)

        inputs2 = keras.Input(shape=(self.inputlength+len(self.listLog[1]),))
        #_, GrpNet = build_LSTM(inputs2, len(self.listLog[0]), "placeGripperScope", "placeGripperOut", n_layers=self.hlayers_place, size=self.size_place, reg=self.reg)
        GrpNet = build_Sub(inputs2, len(self.listLog[0]), "placeGripperScope", "placeGripperOut", n_layers=self.hlayers_place, size=self.size_place, reg=self.reg0)

        inputs3 = keras.Input(shape=(self.inputlength+len(self.listLog[1]),))
        #_, TabNet = build_LSTM(inputs3, len(self.listLog[2]), "placeTableScope", "placeTableOut", n_layers=self.hlayers_place, size=self.size_place, reg=self.reg)
        TabNet = build_Sub(inputs3, len(self.listLog[2]), "placeTableScope", "placeTableOut", n_layers=self.hlayers_place, size=self.size_place, reg=self.reg0)


        modelPlaceObj = keras.models.Model(inputs=inputs1, outputs=ObjNet, name="placeObjeNet")
        modelPlaceGrp = keras.models.Model(inputs=inputs2, outputs=GrpNet, name="placeGrpNet")
        modelPlaceTab = keras.models.Model(inputs=inputs3, outputs=TabNet, name="placeTabNet")

        return modelPlaceObj, modelPlaceGrp, modelPlaceTab

    #------------------Train and save Model---------------
    #----TODO train, adapt input etc

    def reshapeInput(self, path_rai, model_dir):

        dataInput=np.load(path_rai+'/dataset/'+model_dir+'Input.npy')
        dataInstruct=np.load(path_rai+'/dataset/'+model_dir+'Instruction.npy')
        dataLogicals=np.load(path_rai+'/dataset/'+model_dir+'Logicals.npy')

        idx=list(range(dataInput.shape[0]))
        random.shuffle(idx)

        dataInput=dataInput[idx,:]
        dataInstruct=dataInstruct[idx,:]
        dataLogicals=dataLogicals[idx,:]

        #-------------------------------------------
        #----grasp gripper object

        graspIn=dataInput[dataInstruct[:,0]==1,:]
        graspFinalOut=dataLogicals[dataInstruct[:,0]==1,:].astype('int')

        #-------------------------------------------
        #----place gripper object table

        placeIn=dataInput[dataInstruct[:,1]==1,:]
        placeFinalOut=dataLogicals[dataInstruct[:,1]==1,:].astype('int')


        #return dataInput, dataInstruct, graspFinalIn.reshape(-1,2,graspIn.shape[1]+2*self.numLogicals), graspFinalOut.reshape(-1,2,self.numLogicals), placeFinalIn.reshape(-1,3,placeIn.shape[1]+3*self.numLogicals), placeFinalOut.reshape(-1,3,self.numLogicals)
        return dataInput, dataInstruct, graspIn, graspFinalOut, placeIn, placeFinalOut
        

    def step_decay(self,epoch):
        initial_lrate = self.lr
        drop = self.lr_drop
        epochs_drop = self.epoch_drop
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate


    def train(self, path_rai, model_dir, saveToFile=True):
        if not model_dir=='':
            if self.mode==1:
                model_dir=model_dir+'/'
            elif self.mode==2:
                model_dir=model_dir+'_mixed2/'
            elif self.mode==0:
                model_dir=model_dir+'_mixed0/'

        dataInput, dataInstruct, graspFinalIn, graspFinalOut, placeFinalIn, placeFinalOut=self.reshapeInput(path_rai, model_dir)
        #print(np.any(np.isnan(dataInput)))
        #print(np.any(np.isnan(dataInstruct)))
        #print(np.any(np.isnan(graspFinalIn)))
        #print(np.any(np.isnan(graspFinalOut)))
        #print(np.any(np.isnan(placeFinalIn)))
        #print(np.any(np.isnan(placeFinalOut)))

        tf.keras.backend.get_session().run(tf.global_variables_initializer())

        tbInstruction = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/Instruction')
        tbGraspObj = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/GraspObj')
        tbGraspGrp = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/GraspGrp')

        tbPlaceObj = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/PlaceObj')
        tbPlaceGrp = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/PlaceGrp')
        tbPlaceTab = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/PlaceTab')

        if not(os.path.exists(path_rai+'/logs/'+self.timestamp+'/tmp')):
                os.makedirs(path_rai+'/logs/'+self.timestamp+'/tmp')

        saveInst=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelInstruct.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        savegraspObj=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelGraspObj.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        savegraspGrp=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelGraspGrp.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)

        saveplaceObj=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelPlaceObj.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        saveplaceGrp=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelPlaceGrp.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        saveplaceTab=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelPlaceTab.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)

        if saveToFile:
            if not(os.path.exists(path_rai+'/logs/'+self.timestamp)):
                os.makedirs(path_rai+'/logs/'+self.timestamp)

            with open(path_rai+'/logs/'+self.timestamp+'/params.txt', 'w') as f:
                for item in self.params:
                    for i in item:
                        f.write(i+"\t".expandtabs(4))
                    f.write("\n")
            
            shutil.copyfile(path_rai+'/dataset/'+model_dir+'Sets.txt',path_rai+'/logs/'+self.timestamp+'/'+model_dir[:15]+'-Sets.txt')


        print("------Train Network. "+estimateT(self.epochs_inst)+" + "+estimateT(self.epochs_grasp+self.epochs_place, num=5)+"------")
        print("Train Instruct for "+str(self.epochs_inst)+" epochs (hidden layers "+str(self.hlayers_inst)+", size "+str(self.size_inst)+"). "+estimateT(self.epochs_inst))
        modelInstructHist=self.modelInstruct.fit(x=dataInput, y={"instructOut": dataInstruct}, batch_size=32, epochs=self.epochs_inst,
                                                shuffle=True, verbose=0, validation_split=self.val_split,
                                                callbacks=[tbInstruction,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),saveInst, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])

        print("Train Grasp for "+str(self.epochs_grasp)+" epochs (hidden layers "+str(self.hlayers_grasp)+", size "+str(self.size_grasp)+"). "+estimateT(self.epochs_grasp, num=2))
        print("GraspObj")
        modelGraspObjHist=self.modelGraspObj.fit(x=graspFinalIn, y={"graspObjectOut":graspFinalOut[:,0,self.listLog[1]]}, batch_size=32,
                                        epochs=self.epochs_grasp, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbGraspObj,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),savegraspObj, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])
        print("GraspGrp")
        modelGraspGrpHist=self.modelGraspGrp.fit(x=np.concatenate((graspFinalIn, graspFinalOut[:,0,self.listLog[1]]), axis=1), y={"graspGripperOut":graspFinalOut[:,1,self.listLog[0]]}, batch_size=32,
                                        epochs=self.epochs_grasp, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbGraspGrp,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),savegraspGrp, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])

        print("Train Place for "+str(self.epochs_place)+" epochs (hidden layers "+str(self.hlayers_place)+", size "+str(self.size_place)+"). "+estimateT(self.epochs_place, num=3))      
        print("PlaceObj")
        modelPlaceObjHist=self.modelPlaceObj.fit(x=placeFinalIn, y={"placeObjectOut": placeFinalOut[:,0,self.listLog[1]]},batch_size=32,
                                        epochs=self.epochs_place, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbPlaceObj,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),saveplaceObj, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])
        print("PlaceGrp")
        modelPlaceGrpHist=self.modelPlaceGrp.fit(x=np.concatenate((placeFinalIn, placeFinalOut[:,0,self.listLog[1]]), axis=1), y={"placeGripperOut": placeFinalOut[:,1,self.listLog[0]]},batch_size=32,
                                        epochs=self.epochs_place, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbPlaceGrp,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),saveplaceGrp, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])
        print("PlaceTab")
        modelPlaceTabHist=self.modelPlaceTab.fit(x=np.concatenate((placeFinalIn, placeFinalOut[:,0,self.listLog[1]]), axis=1), y={"placeTableOut": placeFinalOut[:,2,self.listLog[2]]},batch_size=32,
                                        epochs=self.epochs_place, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbPlaceTab,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),saveplaceTab, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split, patience=70)])


        if saveToFile:
            self.modelInstruct.save(path_rai+'/logs/'+self.timestamp+'/modelInstruct.h5')
            self.modelGraspObj.save(path_rai+'/logs/'+self.timestamp+'/modelGraspObj.h5')
            self.modelGraspGrp.save(path_rai+'/logs/'+self.timestamp+'/modelGraspGrp.h5')
            self.modelPlaceObj.save(path_rai+'/logs/'+self.timestamp+'/modelPlaceObj.h5')
            self.modelPlaceGrp.save(path_rai+'/logs/'+self.timestamp+'/modelPlaceGrp.h5')
            self.modelPlaceTab.save(path_rai+'/logs/'+self.timestamp+'/modelPlaceTab.h5')
            shutil.rmtree(path_rai+'/logs/'+self.timestamp+'/tmp', ignore_errors=True)

        return modelInstructHist, modelGraspObjHist, modelGraspGrpHist, modelPlaceObjHist, modelPlaceGrpHist, modelPlaceTabHist

#--------------------------------------------------------

class ClassifierChainLSTM():
    def __init__(self): #no itialization -> either load or build
        pass

    def load_net(self,path_rai, model_dir=''):
        if not model_dir=='':
            model_dir=model_dir+'_3d/'
        self.modelInstruct=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelInstruct.h5')
        self.modelGrasp=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelGrasp.h5')
        self.modelPlace=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelPlace.h5')


    def build_net(self,
                inputlength,
                numInstruct,
                numLogicals,
                epochs_inst=300,
                n_layers_inst=2, 
                size_inst=200,
                epochs_grasp=300,
                n_layers_grasp=1,
                size_grasp=200,
                epochs_place=300,
                n_layers_place=1,
                size_place=200,
                timestamp="",
                lr=0.001,
                lr_drop=0.5,
                epoch_drop=100,
                clipnorm=1.,
                val_split=0.2,
                reg=0.001
                ):

        self.inputlength=inputlength
        self.numInstruct=numInstruct
        self.numLogicals=numLogicals

        self.epochs_inst=epochs_inst
        self.hlayers_inst=n_layers_inst
        self.size_inst=size_inst

        self.epochs_inst=epochs_inst
        self.hlayers_inst=n_layers_inst
        self.size_inst=size_inst

        self.epochs_grasp=epochs_grasp
        self.hlayers_grasp=n_layers_grasp
        if self.hlayers_grasp<1:
            self.hlayers_grasp=1
        self.size_grasp=size_grasp

        self.epochs_place=epochs_place
        self.hlayers_place=n_layers_place
        if self.hlayers_place<1:
            self.hlayers_place=1
        self.size_place=size_place

        self.lr=lr
        self.lr_drop=lr_drop
        self.epoch_drop=epoch_drop
        self.val_split=val_split
        self.reg=reg

        self.params=[]
        param=["learning rate"]
        param.append("init: "+str(self.lr))
        param.append("drop: "+str(self.lr_drop))
        param.append("epoch_drop: "+str(self.epoch_drop))
        param.append("clipnorm: "+str(clipnorm))
        param.append("val_split: "+str(self.val_split))
        self.params.append(param)
        
        if timestamp=="":
            now=datetime.datetime.now()
            self.timestamp=str(now.year)+str(now.month).zfill(2)+str(now.day).zfill(2)+"-"+str(now.hour).zfill(2)+str(now.minute).zfill(2)+str(now.second).zfill(2)+"_3d"
        else:
            self.timestamp=timestamp

        self.modelInstruct = self.build_Instruct()
        self.modelGrasp = self.build_Grasp()
        self.modelPlace = self.build_Place()

        self.modelInstruct.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy") #decay=0.001/self.epochs_inst
        self.modelGrasp.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_grasp
        self.modelPlace.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_grasp

    #------------------build subnetwork---------------

    def build_Instruct(self):
        # One Hot : Multi label classification
        param=["Instruct"]
        param.append("hlayers_inst: "+str(self.hlayers_inst))
        param.append("size_inst: "+str(self.size_inst))
        param.append("epochs_inst: "+str(self.epochs_inst))
        self.params.append(param)

        inputs = keras.Input(shape=(None,self.inputlength))
        #_, InstructNet = build_LSTM(inputs, self.numInstruct, "instructScope", "instructOut", n_layers=self.hlayers_inst, size=self.size_inst, reg=self.reg)
        _, InstructNet = build_LSTM(inputs, self.size_inst, "instructScope", "instructMid", n_layers=1, size=self.size_inst, reg=self.reg)
        
        InstructNet = build_Sub(InstructNet, self.numInstruct, "instructScope", "instructOut", n_layers=self.hlayers_inst, size=self.size_inst)

        modelInstuct = keras.models.Model(inputs=inputs, outputs=[InstructNet], name="instructionNet")
        return modelInstuct

    def build_Grasp(self):
        # Multi Hot: Multi class classification through 2 steps
        param=["Grasp"]
        param.append("hlayers_grasp: "+str(self.hlayers_grasp))
        param.append("size_grasp: "+str(self.size_grasp))
        param.append("epochs_grasp: "+str(self.epochs_grasp))
        self.params.append(param)
        
        inputs = keras.Input(shape=(None,self.inputlength+self.numLogicals))
        #_, GraspNet = build_LSTM(inputs, self.numLogicals, "graspScope", "graspOut", n_layers=self.hlayers_grasp, size=self.size_grasp, reg=self.reg)
        _, GraspNet = build_LSTM(inputs, self.size_grasp, "graspScope", "graspMid", n_layers=1, size=self.size_grasp, reg=self.reg)
        
        GraspNet = build_Sub(GraspNet, self.numLogicals, "graspScope", "graspOut", n_layers=self.hlayers_grasp, size=self.size_grasp)

        modelGrasp = keras.models.Model(inputs=inputs, outputs=GraspNet, name="graspObjNet")

        return modelGrasp

    def build_Place(self):
        # Multi Hot: Multi class classification through 3 steps
        param=["Place"]
        param.append("hlayers_place: "+str(self.hlayers_place))
        param.append("size_place: "+str(self.size_place))
        param.append("epochs_place: "+str(self.epochs_place))
        self.params.append(param)

        inputs = keras.Input(shape=(None,self.inputlength+2*self.numLogicals))
        #_, PlaceNet = build_LSTM(inputs, self.numLogicals, "placeScope", "placeOut", n_layers=self.hlayers_place, size=self.size_place, reg=self.reg)
        _, PlaceNet = build_LSTM(inputs, self.size_place, "placeScope", "placeMid", n_layers=1, size=self.size_place, reg=self.reg)   

        PlaceNet = build_Sub(PlaceNet, self.numLogicals, "placeScope", "placeOut", n_layers=self.hlayers_place, size=self.size_place)

        modelPlace = keras.models.Model(inputs=inputs, outputs=PlaceNet, name="graspObjNet")

        return modelPlace

    #------------------Train and save Model---------------
    #----TODO train, adapt input etc

    def reshapeInput(self, path_rai, model_dir):

        dataInput=np.load(path_rai+'/dataset/'+model_dir+'Input.npy') #None inputsize
        dataInstruct=np.load(path_rai+'/dataset/'+model_dir+'Instruction.npy') #None numinstr
        dataLogicals=np.load(path_rai+'/dataset/'+model_dir+'Logicals.npy') #None 3 numlog
        dataInstructPrev=np.load(path_rai+'/dataset/'+model_dir+'InstructionPrev.npy') #None 4 3*numinstr
        dataLogicalsPrev=np.load(path_rai+'/dataset/'+model_dir+'LogicalsPrev.npy') #None 3 3 2*numlog

        """ idx=list(range(dataInput.shape[0]))
        random.shuffle(idx)

        dataInput=dataInput[idx,:]
        dataInstruct=dataInstruct[idx,:]
        dataLogicals=dataLogicals[idx,:,:]
        dataInstructPrev=dataInstructPrev[idx,:,:]
        dataLogicalsPrev=dataLogicalsPrev[idx,:,:,:] """

        #-------------------------------------------
        # Instruction

        idxInstr=[[],[],[],[]]

        InstrIn=np.zeros((dataInput.shape[0], dataInstructPrev.shape[1], dataInput.shape[1]))
        for i in range(InstrIn.shape[0]):
            for j in range(InstrIn.shape[1]):
                if np.any(dataInstructPrev[i,j,:]) or j == 0:
                    InstrIn[i,j,:]=dataInput[i-j,:] # only works for no shuffling of idx
                    #InstrIn[i,j,:]=dataInput[i,:]
                    if j==3:
                        idxInstr[3].append(i)
                        #print(dataInstructPrev[i,:j+1,:])
                        #print(dataInstruct[i,:])
                else:
                    idxInstr[j-1].append(i)
                    #print(dataInstructPrev[i,:j,:])
                    #print(dataInstruct[i,:])
                    break

        print(len(idxInstr[0]),len(idxInstr[1]),len(idxInstr[2]),len(idxInstr[3]))
        #input("test")
        #InstrIn=np.tile(dataInput[:,None,:], [1,4,1])

        InstrIn =np.concatenate((InstrIn, dataInstructPrev), axis=2)
        InstrOut=dataInstruct

        InstrList = [[InstrIn[idxInstr[0], 0:1,:], InstrOut[idxInstr[0],:]],
                     [InstrIn[idxInstr[1], 0:2,:], InstrOut[idxInstr[1],:]],
                     [InstrIn[idxInstr[2], 0:3,:], InstrOut[idxInstr[2],:]],
                     [InstrIn[idxInstr[3], 0:4,:], InstrOut[idxInstr[3],:]]]

        #----grasp gripper object
        #out: 3d -> 2d, prev: 4d-> 3d 
        graspIn  = dataInput[dataInstruct[:,0]==1,:]
        graspOut = dataLogicals[dataInstruct[:,0]==1,:,:].astype('int')
        graspPrev= dataLogicalsPrev[dataInstruct[:,0]==1,:,:,:]

        graspIn  = np.repeat(graspIn,2,axis=0)

        graspPrev=np.reshape(graspPrev, (graspPrev.shape[0]*3, 3,graspPrev.shape[3]))
        graspPrev=np.delete(graspPrev, list(range(2, graspPrev.shape[0],3)), axis=0)
        graspPrev=graspPrev[:,:,:-7]

        graspOut =np.reshape(graspOut,(graspOut.shape[0]*3, graspOut.shape[2]))
        graspOut =np.delete(graspOut, list(range(2, graspOut.shape[0],3)), axis=0)

        idxGrasp = [[],[]]

        graspInFinal=np.zeros((graspIn.shape[0], graspPrev.shape[1], graspIn.shape[1]))
        for i in range(graspInFinal.shape[0]):
            for j in range(graspInFinal.shape[1]):
                if np.any(graspPrev[i,j,:]) or j == 0:
                    graspInFinal[i,j,:]=graspIn[i,:]
                    if j==2:
                        break
                else:
                    idxGrasp[j-1].append(i)
                    #print(graspPrev[i,0:j,:])
                    #print(graspOut[i,:])
                    break

        print(len(idxGrasp[0]), len(idxGrasp[1]))
        input("test")

        #graspInFinal=np.tile(graspIn[:,None,:], [1,3,1])

        graspInFinal=np.concatenate((graspInFinal, graspPrev), axis=2)
        graspInFinal=graspInFinal[:,0:2,:]
        
        graspList = [[graspInFinal[idxGrasp[0],0:1,:], graspOut[idxGrasp[0],:]],
                     [graspInFinal[idxGrasp[1],0:2,:], graspOut[idxGrasp[1],:]]]

        #-------------------------------------------
        #----place gripper object table

        placeIn  = dataInput[dataInstruct[:,1]==1,:]
        placeOut = dataLogicals[dataInstruct[:,1]==1,:,:].astype('int')
        placePrev= dataLogicalsPrev[dataInstruct[:,1]==1,:,:,:]

        placeIn  = np.repeat(placeIn, 3, axis=0)
        placePrev= np.reshape(placePrev, (placePrev.shape[0]*3, 3, placePrev.shape[3]))
        placeOut = np.reshape(placeOut, (placeOut.shape[0]*3, placeOut.shape[2]))

        idxPlace=[[],[],[]]
        placeFinalIn=np.zeros((placeIn.shape[0], placePrev.shape[1], placeIn.shape[1]))
        for i in range(placeFinalIn.shape[0]):
            for j in range(placeFinalIn.shape[1]):
                if np.any(placePrev[i,j,:]) or j == 0:
                    placeFinalIn[i,j,:]=placeIn[i,:]
                    if j==2:
                        idxPlace[2].append(i)
                        #print(placePrev[i,:j+1,:])
                        #print(placeOut[i,:])
                else:
                    idxPlace[j-1].append(i)
                    #print(placePrev[i,:j,:])
                    #print(placeOut[i,:])
                    break
        
        print(len(idxPlace[0]),len(idxPlace[1]),len(idxPlace[2]))
        #input("test")

        #placeFinalIn = np.tile(placeIn[:,None,:], [1,3,1])
        placeFinalIn=np.concatenate((placeFinalIn, placePrev), axis=2)

        placeList = [[placeFinalIn[idxPlace[0],0:1,:], placeOut[idxPlace[0],:]],
                     [placeFinalIn[idxPlace[1],0:2,:], placeOut[idxPlace[1],:]],
                     [placeFinalIn[idxPlace[2],0:3,:], placeOut[idxPlace[2],:]]]

        #print(InstrIn[0,:,:])
        #print(InstrOut[0,:])
        #print(graspInFinal[0,:,:])
        #print(graspOut[graspOut.any(axis=1),:])
        #print(placeFinalIn[0,:,:])

        #input("test")

        #print(placeOut[placeOut.any(axis=1),:])
        if placeOut[~placeOut.any(axis=1),:].size or graspOut[~graspOut.any(axis=1),:].size:
            print(placeOut[~placeOut.any(axis=1),:])
            print(graspOut[~graspOut.any(axis=1),:])
            input("Zero output in training data")


        #return dataInput, dataInstruct, graspFinalIn.reshape(-1,2,graspIn.shape[1]+2*self.numLogicals), graspFinalOut.reshape(-1,2,self.numLogicals), placeFinalIn.reshape(-1,3,placeIn.shape[1]+3*self.numLogicals), placeFinalOut.reshape(-1,3,self.numLogicals)
        #return InstrIn, InstrOut, graspInFinal, graspOut, placeFinalIn, placeOut
        return InstrList, graspList, placeList

    def step_decay(self,epoch):
        initial_lrate = self.lr
        drop = self.lr_drop
        epochs_drop = self.epoch_drop
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate


    def train(self, path_rai, model_dir, num_batch_it, saveToFile=True):
        #if not model_dir=='':
        #    model_dir=model_dir+'_3d/'

        #dataInput, dataInstruct, graspFinalIn, graspFinalOut, placeFinalIn, placeFinalOut=self.reshapeInput(path_rai, model_dir)
        InstrList, graspList, placeList = self.reshapeInput(path_rai, model_dir)
        #print(np.any(np.isnan(dataInput)))
        #print(np.any(np.isnan(dataInstruct)))
        #print(np.any(np.isnan(graspFinalIn)))
        #print(np.any(np.isnan(graspFinalOut)))
        #print(np.any(np.isnan(placeFinalIn)))
        #print(np.any(np.isnan(placeFinalOut)))

        tf.keras.backend.get_session().run(tf.global_variables_initializer())

        tbInstruction = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/Instruction')
        tbGrasp = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/Grasp')
        tbPlace = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/Place')


        if not(os.path.exists(path_rai+'/logs/'+self.timestamp+'/tmp')):
                os.makedirs(path_rai+'/logs/'+self.timestamp+'/tmp')

        saveInst=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelInstruct.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        savegrasp=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelGraspObj.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        saveplace=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelPlaceObj.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)

        if saveToFile:
            if not(os.path.exists(path_rai+'/logs/'+self.timestamp)):
                os.makedirs(path_rai+'/logs/'+self.timestamp)

            with open(path_rai+'/logs/'+self.timestamp+'/params.txt', 'w') as f:
                for item in self.params:
                    for i in item:
                        f.write(i+"\t".expandtabs(4))
                    f.write("\n")
            
            shutil.copyfile(path_rai+'/dataset/'+model_dir+'Sets.txt',path_rai+'/logs/'+self.timestamp+'/'+model_dir[:15]+'-Sets.txt')


        print("------Train Network. "+estimateT_lstm(self.epochs_grasp+self.epochs_place+self.epochs_inst)+"------")

        print("Train Grasp for "+str(self.epochs_grasp)+" epochs (hidden layers "+str(self.hlayers_grasp)+", size "+str(self.size_grasp)+"). "+estimateT_lstm(self.epochs_grasp, num=1))
        print_it = math.floor(10/num_batch_it)
        idx=list(range(2))
        losses=[]
        for j in range(math.ceil(self.epochs_grasp/num_batch_it)):
            #random.shuffle(idx)
            loss=[0,0]
            val_loss=[0,0]
            for i in idx:
                modelGraspHist=self.modelGrasp.fit(x=graspList[i][0], y={"graspOut":graspList[i][1]}, batch_size=32,
                                        epochs=num_batch_it, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbGrasp,keras.callbacks.LearningRateScheduler(self.step_decay),savegrasp, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])
                loss[i]     = modelGraspHist.history["loss"][-1]
                val_loss[i] = modelGraspHist.history["val_loss"][-1]

            losses.append(sum(val_loss))
            if (j%print_it)==0:
                now=datetime.datetime.now()
                timestamp=str(now.year)+"."+str(now.month).zfill(2)+"."+str(now.day).zfill(2)+" "+str(now.hour).zfill(2)+":"+str(now.minute).zfill(2)+":"+str(now.second).zfill(2)
                print("Epoch "+str(j*num_batch_it).rjust(5)+" ended on "+timestamp+
                      ", loss: ["+" , ".join("{:.6f}".format(l) for l in loss)+"] = "+"{:.6f}".format(sum(loss))+
                      ", val_loss: ["+" , ".join("{:.6f}".format(l) for l in val_loss)+"] = "+"{:.6f}".format(sum(val_loss)))
            if j>math.ceil(50/num_batch_it):
                if losses[j]>losses[j-math.ceil(50/num_batch_it)]-0.0001:
                    print(str(losses[j])+" > "+ str(losses[j-math.ceil(50/num_batch_it)]))
                    break


        print("Train Place for "+str(self.epochs_place)+" epochs (hidden layers "+str(self.hlayers_place)+", size "+str(self.size_place)+"). "+estimateT_lstm(self.epochs_place, num=3))      
        idx=list(range(3))
        losses=[]
        for j in range(math.ceil(self.epochs_place/num_batch_it)):
            #random.shuffle(idx)
            loss=[0,0,0]
            val_loss=[0,0,0]
            for i in idx:
                modelPlaceHist=self.modelPlace.fit(x=placeList[i][0], y={"placeOut": placeList[i][1]},batch_size=32,
                                        epochs=num_batch_it, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbPlace,keras.callbacks.LearningRateScheduler(self.step_decay),saveplace, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])
                loss[i]     = modelPlaceHist.history["loss"][-1]
                val_loss[i] = modelPlaceHist.history["val_loss"][-1]

            losses.append(sum(val_loss))
            if (j%print_it)==0:
                now=datetime.datetime.now()
                timestamp=str(now.year)+"."+str(now.month).zfill(2)+"."+str(now.day).zfill(2)+" "+str(now.hour).zfill(2)+":"+str(now.minute).zfill(2)+":"+str(now.second).zfill(2)
                print("Epoch "+str(j*num_batch_it).rjust(5)+" ended on "+timestamp+
                      ", loss: ["+" , ".join("{:.6f}".format(l) for l in loss)+"] = "+"{:.6f}".format(sum(loss))+
                      ", val_loss: ["+" , ".join("{:.6f}".format(l) for l in val_loss)+"] = "+"{:.6f}".format(sum(val_loss)))
            if j>math.ceil(50/num_batch_it):
                if losses[j]>losses[j-math.ceil(50/num_batch_it)]-0.0001:
                    print(str(losses[j])+" > "+ str(losses[j-math.ceil(50/num_batch_it)]))
                    break


        print("Train Instruct for "+str(self.epochs_inst)+" epochs (hidden layers "+str(self.hlayers_inst)+", size "+str(self.size_inst)+"). "+estimateT_lstm(self.epochs_inst))
        idx=list(range(4))
        losses=[]
        for j in range(math.ceil(self.epochs_inst/num_batch_it)):
            #random.shuffle(idx)
            loss=[0,0,0,0]
            val_loss=[0,0,0,0]
            for i in idx:
                modelInstructHist=self.modelInstruct.fit(x=InstrList[0][0], y={"instructOut": InstrList[0][1]}, batch_size=32,
                                        epochs=num_batch_it, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbInstruction,keras.callbacks.LearningRateScheduler(self.step_decay),saveInst, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])
                loss[i]     = modelInstructHist.history["loss"][-1]
                val_loss[i] = modelInstructHist.history["val_loss"][-1]
            losses.append(sum(val_loss))
            if (j%print_it)==0:
                now=datetime.datetime.now()
                timestamp=str(now.year)+"."+str(now.month).zfill(2)+"."+str(now.day).zfill(2)+" "+str(now.hour).zfill(2)+":"+str(now.minute).zfill(2)+":"+str(now.second).zfill(2)
                print("Epoch "+str(j*num_batch_it).rjust(5)+" ended on "+timestamp+
                      ", loss: ["+" , ".join("{:.6f}".format(l) for l in loss)+"] = "+"{:.6f}".format(sum(loss))+
                      ", val_loss: ["+" , ".join("{:.6f}".format(l) for l in val_loss)+"] = "+"{:.6f}".format(sum(val_loss)))
            if j>math.ceil(50/num_batch_it):
                if losses[j]>losses[j-math.ceil(50/num_batch_it)]-0.0001:
                    print(str(losses[j])+" > "+ str(losses[j-math.ceil(50/num_batch_it)]))
                    break

        if saveToFile:
            self.modelInstruct.save(path_rai+'/logs/'+self.timestamp+'/modelInstruct.h5')
            self.modelGrasp.save(path_rai+'/logs/'+self.timestamp+'/modelGrasp.h5')
            self.modelPlace.save(path_rai+'/logs/'+self.timestamp+'/modelPlace.h5')
            shutil.rmtree(path_rai+'/logs/'+self.timestamp+'/tmp', ignore_errors=True)

        return modelInstructHist, modelGraspHist, modelPlaceHist

#---------------------------- Mixed ----------------------------------------
class ClassifierMixed():
    def __init__(self): #no itialization -> either load or build
        pass

    def load_net(self,path_rai, model_dir=''):
        if not model_dir=='':
            if self.mode==1:
                model_dir=model_dir+'_mixed/'
            elif self.mode==2:
                model_dir=model_dir+'_mixed2/'
            elif self.mode==0:
                model_dir=model_dir+'_mixed0/'
            elif self.mode==3:
                model_dir=model_dir+'_mixed3/'
            elif self.mode==11:
                model_dir=model_dir+'_mixed11/'
            elif self.mode==12:
                model_dir=model_dir+'_mixed12/'

        if os.path.isfile(path_rai+'/logs/'+model_dir+'modelInstruct_tmp.h5') and False:
            self.modelInstruct=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelInstruct_tmp.h5')
        else:
            self.modelInstruct=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelInstruct.h5')
        self.modelGraspObj=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelGraspObj.h5')
        self.modelGraspGrp=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelGraspGrp.h5')

        self.modelPlaceObj=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelPlaceObj.h5')
        self.modelPlaceGrpTab=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelPlaceGrpTab.h5')
        #self.modelPlaceTab=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelPlaceTab.h5')


    def build_net(self,
                inputlength,
                numInstruct,
                numLogicals,
                epochs_inst=300,
                n_layers_inst=2, 
                size_inst=200,
                epochs_grasp=300,
                n_layers_grasp=1,
                size_grasp=200,
                epochs_place=300,
                n_layers_place=1,
                size_place=200,
                timestamp="",
                lr=0.001,
                lr_drop=0.5,
                epoch_drop=100,
                clipnorm=1.,
                val_split=0.2,
                reg=0.000001,
                reg0=0.001,
                listLog=[],
                num_batch_it=1,
                mode=1,
                n_layers_inst2=0
                ):

        self.inputlength=inputlength
        self.numInstruct=numInstruct
        self.numLogicals=numLogicals

        self.epochs_inst=epochs_inst
        self.hlayers_inst=n_layers_inst
        self.size_inst=size_inst

        self.hlayers_inst2=n_layers_inst2


        self.epochs_inst=epochs_inst
        self.hlayers_inst=n_layers_inst
        self.size_inst=size_inst

        self.epochs_grasp=epochs_grasp
        self.hlayers_grasp=n_layers_grasp
        if self.hlayers_grasp<1:
            self.hlayers_grasp=1
        self.size_grasp=size_grasp

        self.epochs_place=epochs_place
        self.hlayers_place=n_layers_place
        if self.hlayers_place<1:
            self.hlayers_place=1
        self.size_place=size_place

        self.lr=lr
        self.lr_drop=lr_drop
        self.epoch_drop=epoch_drop
        self.val_split=val_split
        self.reg=reg
        self.reg0=reg0
        self.mode=mode

        self.listLog=listLog
        if self.listLog is []:
            for _ in range(3):
                self.listLog.append(list(range(numLogicals)))

        print(self.listLog)

        self.params=[]
        param=["learning rate"]
        param.append("init: "+str(self.lr))
        param.append("drop: "+str(self.lr_drop))
        param.append("epoch_drop: "+str(self.epoch_drop))
        param.append("clipnorm: "+str(clipnorm))
        param.append("reg_l2: "+str(reg0)+"_"+str(reg))
        param.append("val_split: "+str(self.val_split))
        param.append("num_batch_it: "+str(num_batch_it))
        self.params.append(param)
        
        if timestamp=="":
            now=datetime.datetime.now()
            self.timestamp=str(now.year)+str(now.month).zfill(2)+str(now.day).zfill(2)+"-"+str(now.hour).zfill(2)+str(now.minute).zfill(2)+str(now.second).zfill(2)+"_mixed"
            if mode==2:
                self.timestamp=self.timestamp+"2"
            elif mode==0:
                self.timestamp=self.timestamp+"0"
            elif mode==3:
                self.timestamp=self.timestamp+"3"
            elif mode==11:
                self.timestamp=self.timestamp+"11"
            elif mode==12:
                self.timestamp=self.timestamp+"12"

        else:
            self.timestamp=timestamp

        self.modelInstruct = self.build_Instruct()
        self.modelGraspObj, self.modelGraspGrp = self.build_Grasp()
        self.modelPlaceObj, self.modelPlaceGrpTab = self.build_Place()

        self.modelInstruct.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy") #decay=0.001/self.epochs_inst
        self.modelGraspObj.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_grasp
        self.modelGraspGrp.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_grasp

        self.modelPlaceObj.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_place
        self.modelPlaceGrpTab.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_place

    #------------------build subnetwork---------------

    def build_Instruct(self):
        # One Hot : Multi label classification
        param=["Instruct"]
        param.append("hlayers_inst: "+str(self.hlayers_inst2)+"_"+str(self.hlayers_inst))
        param.append("size_inst: "+str(self.size_inst))
        param.append("epochs_inst: "+str(self.epochs_inst))
        self.params.append(param)

        inputs = keras.Input(shape=(None,self.inputlength))
        if self.hlayers_inst<0:
            print("Pure LSTM")
            _, InstructNet = build_LSTM(inputs, self.numInstruct, "instructScope", "instructOut", n_layers=self.hlayers_inst2, size=self.size_inst, reg=self.reg, output_activation=keras.activations.softmax)
        else:    
            _, InstructNet = build_LSTM(inputs, self.size_inst, "instructScope", "instructMid", n_layers=self.hlayers_inst2, size=self.size_inst, reg=self.reg, output_activation=keras.activations.relu)

            InstructNet = build_Sub(InstructNet, self.numInstruct, "instructScope", "instructOut", n_layers=self.hlayers_inst, size=self.size_inst, reg=self.reg0)

        modelInstuct = keras.models.Model(inputs=inputs, outputs=InstructNet, name="instructionNet")
        return modelInstuct

    def build_Grasp(self):
        # Multi Hot: Multi class classification through 2 steps
        param=["Grasp"]
        param.append("hlayers_grasp: "+str(self.hlayers_grasp))
        param.append("size_grasp: "+str(self.size_grasp))
        param.append("epochs_grasp: "+str(self.epochs_grasp))
        self.params.append(param)
        
        #inputs = keras.Input(shape=(2,self.inputlength+2*self.numLogicals))
        inputs1 = keras.Input(shape=(self.inputlength,))
        #_, ObjNet = build_LSTM(inputs1, len(self.listLog[1]), "graspObjectScope", "graspObjectOut", n_layers=self.hlayers_grasp, size=self.size_grasp, reg=self.reg)
        ObjNet = build_Sub(inputs1, len(self.listLog[1]), "graspObjectScope", "graspObjectOut", n_layers=self.hlayers_grasp, size=self.size_grasp, reg=self.reg0)

        inputs2 = keras.Input(shape=(self.inputlength+len(self.listLog[1]),))
        #_, GrpNet = build_LSTM(inputs2, len(self.listLog[0]), "graspGripperScope", "graspGripperOut", n_layers=self.hlayers_grasp, size=self.size_grasp, reg=self.reg)
        GrpNet = build_Sub(inputs2, len(self.listLog[0]), "graspGripperScope", "graspGripperOut", n_layers=self.hlayers_grasp, size=self.size_grasp, reg=self.reg0)


        modelGraspObj = keras.models.Model(inputs=inputs1, outputs=ObjNet, name="graspObjNet")
        modelGraspGrp = keras.models.Model(inputs=inputs2, outputs=GrpNet, name="graspGrpNet")

        return modelGraspObj, modelGraspGrp

    def build_Place(self):
        # Multi Hot: Multi class classification through 3 steps
        param=["Place"]
        param.append("hlayers_place: "+str(self.hlayers_place))
        param.append("size_place: "+str(self.size_place))
        param.append("epochs_place: "+str(self.epochs_place))
        self.params.append(param)

        #inputs = keras.Input(shape=(3,self.inputlength+3*self.numLogicals))
        inputs1 = keras.Input(shape=(self.inputlength,))
        ObjNet = build_Sub(inputs1, len(self.listLog[1]), "placeObjectScope", "placeObjectOut", n_layers=self.hlayers_place, size=self.size_place, reg=self.reg0)

        inputs2 = keras.Input(shape=(self.inputlength+len(self.listLog[1]),))
        GrpNet = build_Sub(inputs2, len(self.listLog[0]), "placeGripperScope", "placeGripperOut", n_layers=self.hlayers_place, size=self.size_place, reg=self.reg0)

        #inputs3 = keras.Input(shape=(self.inputlength+self.numLogicals,))
        TabNet = build_Sub(inputs2, len(self.listLog[2]), "placeTableScope", "placeTableOut", n_layers=self.hlayers_place, size=self.size_place, reg=self.reg0)


        modelPlaceObj = keras.models.Model(inputs=inputs1, outputs=ObjNet, name="placeObjeNet")
        #modelPlaceGrp = keras.models.Model(inputs=inputs2, outputs=GrpNet, name="placeGrpNet")
        #modelPlaceTab = keras.models.Model(inputs=inputs3, outputs=TabNet, name="placeTabNet")
        modelPlaceGrpTab = keras.models.Model(inputs=inputs2, outputs=[GrpNet, TabNet], name="placeGrpTabNet")


        return modelPlaceObj, modelPlaceGrpTab

    #------------------Train and save Model---------------
    #----TODO train, adapt input etc

    def reshapeInput(self, path_rai, model_dir):
        if self.mode in [11,12]:
            Dappend="_new"
        else:
            Dappend=""

        if self.mode == 12:
            NPappend="_feat"
        else:
            NPappend=""

        dataInput=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'Input'+NPappend+'.npy') #None inputsize
        dataInstruct=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'Instruction.npy') #None numinstr
        dataLogicals=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'Logicals.npy') #None 3 numlog

        if os.path.isfile(path_rai+'/dataset'+Dappend+'/'+model_dir+'InputPrev'+NPappend+'.npy'):
            dataInputPrev=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'InputPrev'+NPappend+'.npy') #None 4 3
        else:
            dataInputPrev=np.concatenate((np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'InputPrev1'+NPappend+'.npy'),np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'InputPrev2'+NPappend+'.npy')), axis=0)
            if os.path.isfile(path_rai+'/dataset'+Dappend+'/'+model_dir+'InputPrev3'+NPappend+'.npy'):
                dataInputPrev=np.concatenate((dataInputPrev,np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'InputPrev3'+NPappend+'.npy')), axis=0)
        #dataLogicalsPrev=np.load(path_rai+'/dataset/'+model_dir+'LogicalsPrev.npy') #None 3 3 2*numlog

        idx=list(range(dataInput.shape[0]))
        random.shuffle(idx)

        dataInput=dataInput[idx,:]
        dataInstruct=dataInstruct[idx,:]
        dataLogicals=dataLogicals[idx,:,:]
        dataInputPrev=dataInputPrev[idx, :,:]
        #dataLogicalsPrev[idx,:,:,:]

        #-------------------------------------------
        idxInstr=[[],[],[],[]]
        for i in range(dataInputPrev.shape[0]):
            for j in range(dataInputPrev.shape[1]):
                if np.any(dataInputPrev[i,j,:]) or j == 0:
                    if j==3:
                        idxInstr[3].append(i)
                else:
                    idxInstr[j-1].append(i)
                    break

        if dataInput.shape[0]<80000:
            tmp1In=np.concatenate((dataInputPrev[idxInstr[2], 0:1,:],dataInputPrev[idxInstr[2], 0:3,:]),axis=1)
            tmp1Out=dataInstruct[idxInstr[2],:]

            tmp2In=np.concatenate((dataInputPrev[idxInstr[2], 1:3,:],np.zeros((len(idxInstr[2]),2,dataInputPrev.shape[2])) ),axis=1)
            tmp2Out=dataInstruct[idxInstr[2],:]

            tmp3In=np.concatenate((dataInputPrev[idxInstr[1], 0:1,:],dataInputPrev[idxInstr[1], 0:2,:], np.zeros((len(idxInstr[1]),1,dataInputPrev.shape[2])) ),axis=1)
            tmp3Out=dataInstruct[idxInstr[1],:]

            idxInstr[3]=idxInstr[3]+list(range(dataInputPrev.shape[0], dataInputPrev.shape[0]+tmp1In.shape[0]))
            idxInstr[1]=idxInstr[1]+list(range(dataInputPrev.shape[0]+tmp1In.shape[0], dataInputPrev.shape[0]+tmp1In.shape[0]+tmp2In.shape[0]))
            idxInstr[2]=idxInstr[2]+list(range(dataInputPrev.shape[0]+tmp1In.shape[0]+tmp2In.shape[0], dataInputPrev.shape[0]+tmp1In.shape[0]+tmp2In.shape[0]+tmp3In.shape[0]))


            dataInputPrev2=np.concatenate((dataInputPrev, tmp1In, tmp2In, tmp3In),axis=0)
            dataInstruct2=np.concatenate((dataInstruct, tmp1Out, tmp2Out, tmp3Out),axis=0)

            idx2=list(range(dataInputPrev2.shape[0]))
            random.shuffle(idx2)
            
            random.shuffle(idxInstr[1])
            random.shuffle(idxInstr[2])
            random.shuffle(idxInstr[3])
        else:
            tmp1In=np.concatenate((dataInputPrev[idxInstr[2], 0:1,:],dataInputPrev[idxInstr[2], 0:3,:]),axis=1)
            tmp1Out=dataInstruct[idxInstr[2],:]

            idxInstr[3]=idxInstr[3]+list(range(dataInputPrev.shape[0], dataInputPrev.shape[0]+tmp1In.shape[0]))

            dataInputPrev2=np.concatenate((dataInputPrev, tmp1In),axis=0)
            dataInstruct2=np.concatenate((dataInstruct, tmp1Out),axis=0)
            idx2=list(range(dataInputPrev2.shape[0]))
            random.shuffle(idx2)

        InstrList = [[dataInputPrev2[idxInstr[0], 0:1,:], dataInstruct2[idxInstr[0],:]],
                     [dataInputPrev2[idxInstr[1], 0:2,:], dataInstruct2[idxInstr[1],:]],
                     [dataInputPrev2[idxInstr[2], 0:3,:], dataInstruct2[idxInstr[2],:]],
                     [dataInputPrev2[idxInstr[3], 0:4,:], dataInstruct2[idxInstr[3],:]],
                     [dataInputPrev2[idx2,:,:], dataInstruct2[idx2,:]]
                     ]
    
        print(len(idxInstr[0]),len(idxInstr[1]),len(idxInstr[2]),len(idxInstr[3]))

        #-------------------------------------------
        #----grasp gripper object

        graspIn=dataInput[dataInstruct[:,0]==1,:]
        graspFinalOut=dataLogicals[dataInstruct[:,0]==1,:,:].astype('int')

        #-------------------------------------------
        #----place gripper object table

        placeIn=dataInput[dataInstruct[:,1]==1,:]
        placeFinalOut=dataLogicals[dataInstruct[:,1]==1,:,:].astype('int')



        #return dataInput, dataInstruct, graspFinalIn.reshape(-1,2,graspIn.shape[1]+2*self.numLogicals), graspFinalOut.reshape(-1,2,self.numLogicals), placeFinalIn.reshape(-1,3,placeIn.shape[1]+3*self.numLogicals), placeFinalOut.reshape(-1,3,self.numLogicals)
        return dataInputPrev, dataInstruct, graspIn, graspFinalOut, placeIn, placeFinalOut, InstrList
        

    def step_decay(self,epoch):
        initial_lrate = self.lr
        drop = self.lr_drop
        epochs_drop = self.epoch_drop
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate


    def train(self, path_rai, model_dir, num_batch_it, saveToFile=True):
        if not model_dir=='':
            Dappend=""
            Setappend=""
            checkpoint=50
            if self.mode==1:
                model_dir=model_dir+'/'
            elif self.mode==2:
                model_dir=model_dir+'_mixed2/'
            elif self.mode==0:
                model_dir=model_dir+'_mixed0/'
            elif self.mode==3:
                model_dir=model_dir+'_mixed3/'
            elif self.mode in [11,12]:
                model_dir=model_dir+"_final/"
                Dappend="_new"
                Setappend="_"+str(self.mode)
                checkpoint=20

        dataInputPrev, dataInstruct, graspFinalIn, graspFinalOut, placeFinalIn, placeFinalOut, InstrList=self.reshapeInput(path_rai, model_dir)
        #print(np.any(np.isnan(dataInput)))
        #print(np.any(np.isnan(dataInstruct)))
        #print(np.any(np.isnan(graspFinalIn)))
        #print(np.any(np.isnan(graspFinalOut)))
        #print(np.any(np.isnan(placeFinalIn)))
        #print(np.any(np.isnan(placeFinalOut)))

        tf.keras.backend.get_session().run(tf.global_variables_initializer())

        tbInstruction = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/Instruction')
        tbGraspObj = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/GraspObj')
        tbGraspGrp = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/GraspGrp')

        tbPlaceObj = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/PlaceObj')
        tbPlaceGrp = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/PlaceGrpTab')
        #tbPlaceTab = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/PlaceTab')

        if not(os.path.exists(path_rai+'/logs/'+self.timestamp+'/tmp')):
                os.makedirs(path_rai+'/logs/'+self.timestamp+'/tmp')

        saveInst=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelInstruct.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=5)
        savegraspObj=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelGraspObj.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        savegraspGrp=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelGraspGrp.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)

        saveplaceObj=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelPlaceObj.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        saveplaceGrp=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelPlaceGrpTab.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        #saveplaceTab=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelPlaceTab.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)

        if saveToFile:
            if not(os.path.exists(path_rai+'/logs/'+self.timestamp)):
                os.makedirs(path_rai+'/logs/'+self.timestamp)

            with open(path_rai+'/logs/'+self.timestamp+'/params.txt', 'w') as f:
                for item in self.params:
                    for i in item:
                        f.write(i+"\t".expandtabs(4))
                    f.write("\n")
            
            shutil.copyfile(path_rai+'/dataset'+Dappend+'/'+model_dir+'Sets.txt',path_rai+'/logs/'+self.timestamp+'/'+model_dir[:15]+'-Sets'+Setappend+'.txt')


        print("------Train Network. "+estimateT(self.epochs_inst)+" + "+estimateT(self.epochs_grasp+self.epochs_place, num=5)+"------")

        print("Train Instruct for "+str(self.epochs_inst)+" epochs (hidden layers "+str(self.hlayers_inst)+", size "+str(self.size_inst)+"). "+estimateT(self.epochs_inst))
        print("Training set size: "+str(dataInputPrev.shape[0]))
        #modelInstructHist=self.modelInstruct.fit(x=dataInputPrev, y={"instructOut": dataInstruct}, batch_size=32, epochs=10,
        #                                        shuffle=True, verbose=0, validation_split=self.val_split,
        #                                        callbacks=[keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),saveInst, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])
        """
        finalIt=False
        print_it = math.floor(10/num_batch_it)
        if print_it <1:
            print_it=1
        idx=list(range(4))
        losses=[]
        terminate=False
        numItDone=int(0.5*self.epochs_inst)
        for j in range(math.ceil(0.5*self.epochs_inst/num_batch_it)):
            if not finalIt:
                random.shuffle(idx)
            else:
                idx=[3,0,2,1]
            loss=[0,0,0,0]
            val_loss=[0,0,0,0]
            for i in idx:
                modelInstructHist=self.modelInstruct.fit(x=InstrList[i][0], y={"instructOut": InstrList[i][1]}, batch_size=32,
                                        epochs=num_batch_it, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[keras.callbacks.LearningRateScheduler(self.step_decay),saveInst, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])
                loss[i]     = modelInstructHist.history["loss"][-1]
                if np.isnan(loss[i]) or np.isinf(loss[i]):
                    terminate=True
                    break
                val_loss[i] = modelInstructHist.history["val_loss"][-1]
            
            if terminate:
                break

            losses.append(sum(val_loss))
            if (j%print_it)==0 or finalIt:
                if finalIt:
                    num_batch_it=num_batch_it_tmp
                now=datetime.datetime.now()
                timestamp=str(now.year)+"."+str(now.month).zfill(2)+"."+str(now.day).zfill(2)+" "+str(now.hour).zfill(2)+":"+str(now.minute).zfill(2)+":"+str(now.second).zfill(2)
                print("Epoch "+str(j*num_batch_it).rjust(5)+" ended on "+timestamp+
                      ", loss: ["+", ".join("{:.6f}".format(l) for l in loss)+"] = "+"{:.6f}".format(sum(loss))+
                      ", val_loss: ["+", ".join("{:.6f}".format(l) for l in val_loss)+"] = "+"{:.6f}".format(sum(val_loss)))
            if finalIt:
                break
            
            if j>=math.ceil(checkpoint/num_batch_it):
                #print(losses)
                #print(str(losses[j])+" >? "+str(losses[j-math.ceil(checkpoint/num_batch_it)]-0.0001))
                if losses[j]>losses[j-math.ceil(checkpoint/num_batch_it)]-0.0001:
                    print(str(losses[j])+" > "+ str(losses[j-math.ceil(checkpoint/num_batch_it)]))
                    numItDone=j*num_batch_it+1
                    finalIt=True
                    num_batch_it_tmp=num_batch_it
                    num_batch_it=1
                    #break
        if saveToFile:
            self.modelInstruct.save(path_rai+'/logs/'+self.timestamp+'/modelInstruct_tmp.h5')"""

        final_losses=[["instruct"],["graspobj"],["graspgrp"],["placeobj"],["placegrptab"]]
        numItDone=-5
        #modelInstructHist=self.modelInstruct.fit(x=InstrList[4][0], y={"instructOut": InstrList[4][1]}, batch_size=32,
        #                                        epochs=5, shuffle=True, verbose=0, validation_split=self.val_split,
        #                                        callbacks=[tbInstruction,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),saveInst, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split, patience=10)])
        #if saveToFile:
        #    self.modelInstruct.save(path_rai+'/logs/'+self.timestamp+'/modelInstruct_tmp2.h5')

        modelInstructHist=self.modelInstruct.fit(x=InstrList[4][0], y={"instructOut": InstrList[4][1]}, batch_size=32,
                                                epochs=self.epochs_inst-numItDone-5, shuffle=True, verbose=0, validation_split=self.val_split,
                                                callbacks=[tbInstruction,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),saveInst, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split, patience=10)])

        final_losses[0].append(modelInstructHist.history["loss"][-1])
        final_losses[0].append(modelInstructHist.history["val_loss"][-1])
        if saveToFile:
            self.modelInstruct.save(path_rai+'/logs/'+self.timestamp+'/modelInstruct.h5')
        
        print("Train Grasp for "+str(self.epochs_grasp)+" epochs (hidden layers "+str(self.hlayers_grasp)+", size "+str(self.size_grasp)+"). "+estimateT(self.epochs_grasp, num=2))
        print("Training set size: "+str(graspFinalIn.shape[0]))

        print("GraspObj")
        modelGraspObjHist=self.modelGraspObj.fit(x=graspFinalIn, y={"graspObjectOut":graspFinalOut[:,0,self.listLog[1]]}, batch_size=32,
                                        epochs=self.epochs_grasp, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbGraspObj,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),savegraspObj, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split, patience=30)])
        
        final_losses[1].append(modelGraspObjHist.history["loss"][-1])
        final_losses[1].append(modelGraspObjHist.history["val_loss"][-1])

        if saveToFile:
            self.modelGraspObj.save(path_rai+'/logs/'+self.timestamp+'/modelGraspObj.h5')

        print("GraspGrp")
        modelGraspGrpHist=self.modelGraspGrp.fit(x=np.concatenate((graspFinalIn, graspFinalOut[:,0,self.listLog[1]]), axis=1), y={"graspGripperOut":graspFinalOut[:,1,self.listLog[0]]}, batch_size=32,
                                        epochs=self.epochs_grasp, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbGraspGrp,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),savegraspGrp, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split, patience=30)])

        final_losses[2].append(modelGraspGrpHist.history["loss"][-1])
        final_losses[2].append(modelGraspGrpHist.history["val_loss"][-1])

        if saveToFile:
            self.modelGraspGrp.save(path_rai+'/logs/'+self.timestamp+'/modelGraspGrp.h5')

        print("Train Place for "+str(self.epochs_place)+" epochs (hidden layers "+str(self.hlayers_place)+", size "+str(self.size_place)+"). "+estimateT(self.epochs_place, num=3))      
        print("Training set size: "+str(placeFinalIn.shape[0]))
        print("PlaceObj")
        modelPlaceObjHist=self.modelPlaceObj.fit(x=placeFinalIn, y={"placeObjectOut": placeFinalOut[:,0,self.listLog[1]]},batch_size=32,
                                        epochs=self.epochs_place, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbPlaceObj,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),saveplaceObj, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split, patience=30)])
        
        final_losses[3].append(modelPlaceObjHist.history["loss"][-1])
        final_losses[3].append(modelPlaceObjHist.history["val_loss"][-1])

        if saveToFile:
            self.modelPlaceObj.save(path_rai+'/logs/'+self.timestamp+'/modelPlaceObj.h5')

        print("PlaceGrpTab")
        modelPlaceGrpHist=self.modelPlaceGrpTab.fit(x=np.concatenate((placeFinalIn, placeFinalOut[:,0,self.listLog[1]]), axis=1), y={"placeGripperOut": placeFinalOut[:,1,self.listLog[0]],"placeTableOut": placeFinalOut[:,2,self.listLog[2]]},batch_size=32,
                                        epochs=self.epochs_place, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbPlaceGrp,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),saveplaceGrp, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split, patience=30)])
        
        final_losses[4].append(modelPlaceGrpHist.history["loss"][-1])
        final_losses[4].append(modelPlaceGrpHist.history["val_loss"][-1])

        if saveToFile:
            self.modelPlaceGrpTab.save(path_rai+'/logs/'+self.timestamp+'/modelPlaceGrpTab.h5')

            with open(path_rai+'/logs/'+self.timestamp+'/SummaryLoss.txt', 'a+') as f:
                for floss in final_losses:
                    f.write(floss[0]+": loss "+str(floss[1])+" | val_loss "+str(floss[2])+"\n")
            
            shutil.rmtree(path_rai+'/logs/'+self.timestamp+'/tmp', ignore_errors=True)


        return modelInstructHist, modelGraspObjHist, modelGraspGrpHist, modelPlaceObjHist, modelPlaceGrpHist

#---------------------------------------------

class ClassifierChainNew():
    def __init__(self): #no itialization -> either load or build
        pass

    def load_net(self,path_rai, model_dir=''):
        if not model_dir=='':
            model_dir=model_dir+'_FFnew/'
        self.modelInstruct=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelInstruct.h5')
        self.modelGraspObj=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelGraspObj.h5')
        self.modelGraspGrp=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelGraspGrp.h5')

        self.modelPlaceObj=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelPlaceObj.h5')
        self.modelPlaceGrpTab=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelPlaceGrpTab.h5')

    def build_net(self,
                inputlength,
                numInstruct,
                numLogicals,
                epochs_inst=300,
                n_layers_inst=2, 
                size_inst=200,
                epochs_grasp=300,
                n_layers_grasp=1,
                size_grasp=200,
                epochs_place=300,
                n_layers_place=1,
                size_place=200,
                timestamp="",
                lr=0.001,
                lr_drop=0.5,
                epoch_drop=100,
                clipnorm=1.,
                val_split=0.2,
                listLog=[],
                mode=1,
                reg0=0.001
                ):

        self.inputlength=inputlength
        self.numInstruct=numInstruct
        self.numLogicals=numLogicals

        self.epochs_inst=epochs_inst
        self.hlayers_inst=n_layers_inst
        self.size_inst=size_inst

        self.epochs_inst=epochs_inst
        self.hlayers_inst=n_layers_inst
        self.size_inst=size_inst

        self.epochs_grasp=epochs_grasp
        self.hlayers_grasp=n_layers_grasp
        if self.hlayers_grasp<1:
            self.hlayers_grasp=1
        self.size_grasp=size_grasp

        self.epochs_place=epochs_place
        self.hlayers_place=n_layers_place
        if self.hlayers_place<1:
            self.hlayers_place=1
        self.size_place=size_place

        self.lr=lr
        self.lr_drop=lr_drop
        self.epoch_drop=epoch_drop
        self.val_split=val_split
        self.reg0=reg0
        self.mode=mode

        self.listLog=listLog
        if self.listLog is []:
            for _ in range(3):
                self.listLog.append(list(range(numLogicals)))

        print(self.listLog)

        self.params=[]
        param=["learning rate"]
        param.append("init: "+str(self.lr))
        param.append("drop: "+str(self.lr_drop))
        param.append("epoch_drop: "+str(self.epoch_drop))
        param.append("clipnorm: "+str(clipnorm))
        param.append("reg_l2: "+str(reg0))
        param.append("val_split: "+str(self.val_split))
        param.append("dataset_mode: "+str(self.mode))
        self.params.append(param)
        
        if timestamp=="":
            now=datetime.datetime.now()
            self.timestamp=str(now.year)+str(now.month).zfill(2)+str(now.day).zfill(2)+"-"+str(now.hour).zfill(2)+str(now.minute).zfill(2)+str(now.second).zfill(2)+"_FFnew"
        else:
            self.timestamp=timestamp

        self.modelInstruct = self.build_Instruct()
        self.modelGraspObj, self.modelGraspGrp = self.build_Grasp()
        self.modelPlaceObj, self.modelPlaceGrpTab = self.build_Place()

        #lossesInstruct={"instructOut": "categorical_crossentropy"}
        #lossesGrasp={"graspGripperOut": "categorical_crossentropy", "graspObjectOut": "categorical_crossentropy"}
        #lossesPlace={"placeGripperOut": "categorical_crossentropy", "placeObjectOut": "categorical_crossentropy", "placeTableOut": "categorical_crossentropy"}

        #weightsInstruct={"instructOut": 1.0}
        #weightsGrasp={"graspGripperOut": 0.5, "graspObjectOut": 0.5}
        #weightsPlace={"placeGripperOut": 0.33, "placeObjectOut": 0.33, "placeTableOut": 0.33}

        self.modelInstruct.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy") #decay=0.001/self.epochs_inst
        self.modelGraspObj.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_grasp
        self.modelGraspGrp.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_grasp

        self.modelPlaceObj.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_place
        self.modelPlaceGrpTab.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="categorical_crossentropy")#decay=0.001/self.epochs_place

    #------------------build subnetwork---------------

    def build_Instruct(self):
        # One Hot : Multi label classification
        param=["Instruct"]
        param.append("hlayers_inst: "+str(self.hlayers_inst))
        param.append("size_inst: "+str(self.size_inst))
        param.append("epochs_inst: "+str(self.epochs_inst))
        self.params.append(param)

        inputs = keras.Input(shape=(self.inputlength,))
        InstructNet = build_Sub(inputs, self.numInstruct, "instructScope", "instructOut", n_layers=self.hlayers_inst, size=self.size_inst, reg=self.reg0)
        
        modelInstuct = keras.models.Model(inputs=inputs, outputs=[InstructNet], name="instructionNet")
        return modelInstuct

    def build_Grasp(self):
        # Multi Hot: Multi class classification through 2 steps
        param=["Grasp"]
        param.append("hlayers_grasp: "+str(self.hlayers_grasp))
        param.append("size_grasp: "+str(self.size_grasp))
        param.append("epochs_grasp: "+str(self.epochs_grasp))
        self.params.append(param)
        
        #inputs = keras.Input(shape=(2,self.inputlength+2*self.numLogicals))
        inputs1 = keras.Input(shape=(self.inputlength,))
        #_, ObjNet = build_LSTM(inputs1, len(self.listLog[1]), "graspObjectScope", "graspObjectOut", n_layers=self.hlayers_grasp, size=self.size_grasp, reg=self.reg)
        ObjNet = build_Sub(inputs1, len(self.listLog[1]), "graspObjectScope", "graspObjectOut", n_layers=self.hlayers_grasp, size=self.size_grasp, reg=self.reg0)

        inputs2 = keras.Input(shape=(self.inputlength+len(self.listLog[1]),))
        #_, GrpNet = build_LSTM(inputs2, len(self.listLog[0]), "graspGripperScope", "graspGripperOut", n_layers=self.hlayers_grasp, size=self.size_grasp, reg=self.reg)
        GrpNet = build_Sub(inputs2, len(self.listLog[0]), "graspGripperScope", "graspGripperOut", n_layers=self.hlayers_grasp, size=self.size_grasp, reg=self.reg0)


        modelGraspObj = keras.models.Model(inputs=inputs1, outputs=ObjNet, name="graspObjNet")
        modelGraspGrp = keras.models.Model(inputs=inputs2, outputs=GrpNet, name="graspGrpNet")

        return modelGraspObj, modelGraspGrp

    def build_Place(self):
        # Multi Hot: Multi class classification through 3 steps
        param=["Place"]
        param.append("hlayers_place: "+str(self.hlayers_place))
        param.append("size_place: "+str(self.size_place))
        param.append("epochs_place: "+str(self.epochs_place))
        self.params.append(param)

        inputs1 = keras.Input(shape=(self.inputlength,))
        ObjNet = build_Sub(inputs1, len(self.listLog[1]), "placeObjectScope", "placeObjectOut", n_layers=self.hlayers_place, size=self.size_place, reg=self.reg0)

        inputs2 = keras.Input(shape=(self.inputlength+len(self.listLog[1]),))
        GrpNet = build_Sub(inputs2, len(self.listLog[0]), "placeGripperScope", "placeGripperOut", n_layers=self.hlayers_place, size=self.size_place, reg=self.reg0)
        TabNet = build_Sub(inputs2, len(self.listLog[2]), "placeTableScope", "placeTableOut", n_layers=self.hlayers_place, size=self.size_place, reg=self.reg0)


        modelPlaceObj = keras.models.Model(inputs=inputs1, outputs=ObjNet, name="placeObjeNet")
        modelPlaceGrpTab = keras.models.Model(inputs=inputs2, outputs=[GrpNet, TabNet], name="placeGrpTabNet")


        return modelPlaceObj, modelPlaceGrpTab

    #------------------Train and save Model---------------
    #----TODO train, adapt input etc

    def reshapeInput(self, path_rai, model_dir):

        if self.mode in [11,12]:
            Dappend="_new"
        else:
            Dappend=""

        if self.mode == 12:
            NPappend="_feat"
        else:
            NPappend=""

        dataInput=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'Input'+NPappend+'.npy') #None inputsize
        dataInstruct=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'Instruction.npy') #None numinstr
        dataLogicals=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'Logicals.npy') #None 3 numlog

        idx=list(range(dataInput.shape[0]))
        random.shuffle(idx)

        dataInput=dataInput[idx,:]
        dataInstruct=dataInstruct[idx,:]
        dataLogicals=dataLogicals[idx,:]

        #-------------------------------------------
        #----grasp gripper object

        graspIn=dataInput[dataInstruct[:,0]==1,:]
        graspFinalOut=dataLogicals[dataInstruct[:,0]==1,:].astype('int')

        #-------------------------------------------
        #----place gripper object table

        placeIn=dataInput[dataInstruct[:,1]==1,:]
        placeFinalOut=dataLogicals[dataInstruct[:,1]==1,:].astype('int')


        #return dataInput, dataInstruct, graspFinalIn.reshape(-1,2,graspIn.shape[1]+2*self.numLogicals), graspFinalOut.reshape(-1,2,self.numLogicals), placeFinalIn.reshape(-1,3,placeIn.shape[1]+3*self.numLogicals), placeFinalOut.reshape(-1,3,self.numLogicals)
        return dataInput, dataInstruct, graspIn, graspFinalOut, placeIn, placeFinalOut
        

    def step_decay(self,epoch):
        initial_lrate = self.lr
        drop = self.lr_drop
        epochs_drop = self.epoch_drop
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate


    def train(self, path_rai, model_dir, saveToFile=True):
        if not model_dir=='':
            Dappend=""
            Setappend=""
            if self.mode==1:
                model_dir=model_dir+'/'
            elif self.mode==2:
                model_dir=model_dir+'_mixed2/'
            elif self.mode==0:
                model_dir=model_dir+'_mixed0/'
            elif self.mode==3:
                model_dir=model_dir+'_mixed3/'
            elif self.mode in [11,12]:
                model_dir=model_dir+"_final/"
                Dappend="_new"
                Setappend="_"+str(self.mode)

        dataInput, dataInstruct, graspFinalIn, graspFinalOut, placeFinalIn, placeFinalOut=self.reshapeInput(path_rai, model_dir)
        #print(np.any(np.isnan(dataInput)))
        #print(np.any(np.isnan(dataInstruct)))
        #print(np.any(np.isnan(graspFinalIn)))
        #print(np.any(np.isnan(graspFinalOut)))
        #print(np.any(np.isnan(placeFinalIn)))
        #print(np.any(np.isnan(placeFinalOut)))

        tf.keras.backend.get_session().run(tf.global_variables_initializer())

        tbInstruction = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/Instruction')
        tbGraspObj = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/GraspObj')
        tbGraspGrp = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/GraspGrp')

        tbPlaceObj = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/PlaceObj')
        tbPlaceGrp = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/PlaceGrpTab')

        if not(os.path.exists(path_rai+'/logs/'+self.timestamp+'/tmp')):
                os.makedirs(path_rai+'/logs/'+self.timestamp+'/tmp')

        saveInst=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelInstruct.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        savegraspObj=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelGraspObj.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        savegraspGrp=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelGraspGrp.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)

        saveplaceObj=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelPlaceObj.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)
        saveplaceGrp=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelPlaceGrpTab.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)

        if saveToFile:
            if not(os.path.exists(path_rai+'/logs/'+self.timestamp)):
                os.makedirs(path_rai+'/logs/'+self.timestamp)

            with open(path_rai+'/logs/'+self.timestamp+'/params.txt', 'w') as f:
                for item in self.params:
                    for i in item:
                        f.write(i+"\t".expandtabs(4))
                    f.write("\n")
            
            shutil.copyfile(path_rai+'/dataset'+Dappend+'/'+model_dir+'Sets.txt',path_rai+'/logs/'+self.timestamp+'/'+model_dir[:15]+'-Sets'+Setappend+'.txt')

        final_losses=[["instruct"],["graspobj"],["graspgrp"],["placeobj"],["placegrptab"]]

        print("------Train Network. "+estimateT(self.epochs_inst)+" + "+estimateT(self.epochs_grasp+self.epochs_place, num=5)+"------")
        print("Train Instruct for "+str(self.epochs_inst)+" epochs (hidden layers "+str(self.hlayers_inst)+", size "+str(self.size_inst)+"). "+estimateT(self.epochs_inst))
        print("Training set size: "+str(dataInput.shape[0]))
        modelInstructHist=self.modelInstruct.fit(x=dataInput, y={"instructOut": dataInstruct}, batch_size=32, epochs=self.epochs_inst,
                                                shuffle=True, verbose=0, validation_split=self.val_split,
                                                callbacks=[tbInstruction,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),saveInst, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])

        final_losses[0].append(modelInstructHist.history["loss"][-1])
        final_losses[0].append(modelInstructHist.history["val_loss"][-1])

        if saveToFile:
            self.modelInstruct.save(path_rai+'/logs/'+self.timestamp+'/modelInstruct.h5')

        print("Train Grasp for "+str(self.epochs_grasp)+" epochs (hidden layers "+str(self.hlayers_grasp)+", size "+str(self.size_grasp)+"). "+estimateT(self.epochs_grasp, num=2))
        print("Training set size: "+str(graspFinalIn.shape[0]))
        print("GraspObj")
        modelGraspObjHist=self.modelGraspObj.fit(x=graspFinalIn, y={"graspObjectOut":graspFinalOut[:,0,self.listLog[1]]}, batch_size=32,
                                        epochs=self.epochs_grasp, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbGraspObj,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),savegraspObj, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])
        
        final_losses[1].append(modelGraspObjHist.history["loss"][-1])
        final_losses[1].append(modelGraspObjHist.history["val_loss"][-1])

        if saveToFile:
            self.modelGraspObj.save(path_rai+'/logs/'+self.timestamp+'/modelGraspObj.h5')
        
        print("GraspGrp")
        modelGraspGrpHist=self.modelGraspGrp.fit(x=np.concatenate((graspFinalIn, graspFinalOut[:,0,self.listLog[1]]), axis=1), y={"graspGripperOut":graspFinalOut[:,1,self.listLog[0]]}, batch_size=32,
                                        epochs=self.epochs_grasp, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbGraspGrp,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),savegraspGrp, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])

        final_losses[2].append(modelGraspGrpHist.history["loss"][-1])
        final_losses[2].append(modelGraspGrpHist.history["val_loss"][-1])

        if saveToFile:
            self.modelGraspGrp.save(path_rai+'/logs/'+self.timestamp+'/modelGraspGrp.h5')

        print("Train Place for "+str(self.epochs_place)+" epochs (hidden layers "+str(self.hlayers_place)+", size "+str(self.size_place)+"). "+estimateT(self.epochs_place, num=3))      
        print("Training set size: "+str(placeFinalIn.shape[0]))
        print("PlaceObj")
        modelPlaceObjHist=self.modelPlaceObj.fit(x=placeFinalIn, y={"placeObjectOut": placeFinalOut[:,0,self.listLog[1]]},batch_size=32,
                                        epochs=self.epochs_place, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbPlaceObj,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),saveplaceObj, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split)])
        
        final_losses[3].append(modelPlaceObjHist.history["loss"][-1])
        final_losses[3].append(modelPlaceObjHist.history["val_loss"][-1])

        if saveToFile:
            self.modelPlaceObj.save(path_rai+'/logs/'+self.timestamp+'/modelPlaceObj.h5')
        
        print("PlaceGrpTab")
        modelPlaceGrpHist=self.modelPlaceGrpTab.fit(x=np.concatenate((placeFinalIn, placeFinalOut[:,0,self.listLog[1]]), axis=1), y={"placeGripperOut": placeFinalOut[:,1,self.listLog[0]],"placeTableOut": placeFinalOut[:,2,self.listLog[2]]},batch_size=32,
                                        epochs=self.epochs_place, shuffle=True, verbose=0, validation_split=self.val_split,
                                        callbacks=[tbPlaceGrp,keras.callbacks.LearningRateScheduler(self.step_decay), printEpoch(),saveplaceGrp, keras.callbacks.TerminateOnNaN(), EarlyStopping(self.val_split, patience=70)])

        final_losses[4].append(modelPlaceGrpHist.history["loss"][-1])
        final_losses[4].append(modelPlaceGrpHist.history["val_loss"][-1])

        if saveToFile:
            self.modelPlaceGrpTab.save(path_rai+'/logs/'+self.timestamp+'/modelPlaceGrpTab.h5')


            with open(path_rai+'/logs/'+self.timestamp+'/SummaryLoss.txt', 'a+') as f:
                for floss in final_losses:
                    f.write(floss[0]+": loss "+str(floss[1])+" | val_loss "+str(floss[2])+"\n")
            
            shutil.rmtree(path_rai+'/logs/'+self.timestamp+'/tmp', ignore_errors=True)

        return modelInstructHist, modelGraspObjHist, modelGraspGrpHist, modelPlaceObjHist, modelPlaceGrpHist

#--------------------------------------------------------

