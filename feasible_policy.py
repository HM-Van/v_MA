import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import Model
from keras.objectives import binary_crossentropy

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as KB

import math
import datetime
import shutil
import os
import random
import rai_policy

dir_file=os.path.abspath(os.path.dirname(__file__))

# Test to implement feasibility network: NOT USED

def build_FeasSub(inputs, 
              output_size,
              scope,
              name,
              n_layers=2, 
              size=200, 
              activation=keras.activations.relu,
              output_activation=keras.activations.sigmoid,
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

def build_FeasLSTM(inputs, 
              output_size,
              scope,
              name,
              n_layers=1, 
              size=200, 
              activation=keras.activations.relu,
              output_activation=keras.activations.sigmoid,
              reg=0.001
              ):
    out = inputs # input (batch_size, timespan, input_dim)
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = keras.layers.LSTM(size, activation=activation, dropout=0.2, return_sequences=True, kernel_regularizer=keras.regularizers.l2(reg), recurrent_regularizer=keras.regularizers.l2(reg), bias_regularizer=keras.regularizers.l2(reg))(out) # output (batch_size, timespan, size)
        final = keras.layers.LSTM(output_size, activation=output_activation, return_sequences=False, kernel_regularizer=keras.regularizers.l2(reg), recurrent_regularizer=keras.regularizers.l2(reg), bias_regularizer=keras.regularizers.l2(reg), name=name)(out) # output (batch_size, timespan, output_size)
        #out = keras.layers.LSTM(output_size, activation=activation, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.001), recurrent_regularizer=keras.regularizers.l2(0.001), bias_regularizer=keras.regularizers.l2(0.001))(out) # output (batch_size, timespan, output_size)

    return out, final



class ClassifierFeas():
    def __init__(self): #no itialization -> either load or build
        pass

    def load_net(self,path_rai, model_dir=''):
        #print(path_rai)
        if not model_dir=='':
            if self.mode==1:
                model_dir=model_dir+'/'
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
            elif self.mode==13:
                model_dir=model_dir+'_mixed13/'
            elif self.mode==14:
                model_dir=model_dir+'_mixed14/'
            else:
                model_dir=model_dir+'/'

        if os.path.isfile(path_rai+'/logs/'+model_dir+'modelFeasible.h5'):
            self.modelFeasible=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelFeasible.h5')
        else:
            self.modelFeasible=None

    def build_net(self,
                goallength,
                statelength,
                numInstruct,
                epochs_feas=300,
                n_layers_feas=2, 
                size_feas=200,
                timestamp="",
                lr=0.001,
                lr_drop=0.5,
                epoch_drop=100,
                clipnorm=1.,
                val_split=0.2,
                mode=3,
                listLog=[],
                reg0=0,
                path_rai=dir_file
                ):
        self.goalEncoder=tf.keras.models.load_model(path_rai+'/logs/encoder/encoderGoal.h5', compile=False)

        self.goallength=goallength
        self.statelength=statelength
        self.numInstruct=numInstruct

        self.epochs_feas=epochs_feas
        self.hlayers_feas=n_layers_feas
        self.size_feas=size_feas

        self.lr=lr
        self.lr_drop=lr_drop
        self.epoch_drop=epoch_drop
        self.val_split=val_split
        self.mode=mode
        self.reg0=reg0

        self.listLog=listLog
        if self.listLog is []:
            for _ in range(3):
                self.listLog.append(list(range(statelength/3)))

        print(self.listLog)

        self.params=[]
        param=["learning rate"]
        param.append("init: "+str(self.lr))
        param.append("drop: "+str(self.lr_drop))
        param.append("epoch_drop: "+str(self.epoch_drop))
        param.append("clipnorm: "+str(clipnorm))
        param.append("val_split: "+str(self.val_split))
        param.append("dataset_mode: "+str(self.mode))
        self.params.append(param)
        
        if timestamp=="":
            now=datetime.datetime.now()
            self.timestamp=str(now.year)+str(now.month).zfill(2)+str(now.day).zfill(2)+"-"+str(now.hour).zfill(2)+str(now.minute).zfill(2)+str(now.second).zfill(2)+"_feasible"
            if mode==11:
                self.timestamp=self.timestamp+"11"
            elif mode==12:
                self.timestamp=self.timestamp+"12"
            elif mode==13:
                self.timestamp=self.timestamp+"13"
            elif mode==14:
                self.timestamp=self.timestamp+"14"
        else:
            self.timestamp=timestamp

        self.modelFeasible = self.build_Feasible()

        self.modelFeasible.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="binary_crossentropy") #decay=0.001/self.epochs_inst
        
    #------------------build subnetwork---------------

    def build_Feasible(self):
        # One Hot : Multi label classification
        param=["Instruct"]
        param.append("hlayers_feas: "+str(self.hlayers_feas))
        param.append("size_feas: "+str(self.size_feas))
        param.append("epochs_feas: "+str(self.epochs_feas))
        self.params.append(param)

        inputs = keras.Input(shape=(self.numInstruct+len(self.listLog[0])+len(self.listLog[1])+len(self.listLog[2]),), name='action')
        inputs0 = keras.Input(shape=(self.goallength,), name='goal')
        inputs1=self.goalEncoder(inputs0)
        inputs2 = keras.Input(shape=(self.statelength,), name='state')
        inputs3 = keras.layers.concatenate([inputs1, inputs2])
        feasActNet = build_FeasSub(keras.layers.concatenate([inputs3, inputs]), 1, "feasActScope", "feasActOut", n_layers=self.hlayers_feas, size=self.size_feas, reg=self.reg0)
        feasSkeNet = build_FeasSub(keras.layers.concatenate([inputs3, inputs]), 1, "feasSkeScope", "feasSkeOut", n_layers=self.hlayers_feas, size=self.size_feas, reg=self.reg0)
        feasAct2Net = build_FeasSub(keras.layers.concatenate([inputs2, inputs]), 1, "feasAct2Scope", "feasAct2Out", n_layers=self.hlayers_feas, size=self.size_feas, reg=self.reg0)

        modelFeasible = keras.models.Model(inputs=[inputs0,inputs2, inputs], outputs=[feasActNet, feasSkeNet, feasAct2Net], name="feasNet")
        return modelFeasible

    #------------------Train and save Model---------------
    #----TODO train, adapt input etc

    def reshapeInput(self, path_rai, model_dir):
        if self.mode in [11,12,13,14] and True:
            Dappend="_new"
        else:
            Dappend=""

        if self.mode in [12,14]:
            NPappend="_feat"
        else:
            NPappend=""

        dataInputFeas=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'Input'+NPappend+'.npy')
        dataFeasible=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'Feasible.npy')
        dataInstr=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'Instruction.npy')
        dataLog=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'Logicals.npy')
        dataInputInFeas=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'InputInfeasible'+NPappend+'.npy')
        dataInFeasible=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'InFeasible.npy')
        dataInstrInfeas=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'InFeasibleInstr.npy')
        dataLogInfeas=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'InFeasibleLog.npy')

        #tmpidx=list((range(dataInFeasible.shape[0])))
        #tmpidx=list((range(dataFeasible.shape[0])))
        #random.shuffle(tmpidx)
        #tmpidx=tmpidx[:(dataFeasible.shape[0]-dataInFeasible.shape[0])]
        #tmpidx=tmpidx[:dataInFeasible.shape[0]]

        infeasible=np.where(~dataFeasible[:,0:1].any(axis=1))[0]
        feasible=np.where(dataFeasible[:,0:1].any(axis=1))[0]
        tmpidx=infeasible
        print(len(tmpidx), dataInFeasible.shape[0])
        if len(tmpidx)<dataInFeasible.shape[0]:
            random.shuffle(feasible)
            tmpidx=np.concatenate((tmpidx,feasible[:int(1.5*dataInFeasible.shape[0])-len(tmpidx)]), axis=0)
            

        FeasIn=np.concatenate((dataInputFeas, dataInstr, dataLog[:,1,self.listLog[0]], dataLog[:,0,self.listLog[1]], dataLog[:,2,self.listLog[2]]), axis=1)
        InFeasIn=np.concatenate((dataInputInFeas, dataInstrInfeas, dataLogInfeas[:,1,self.listLog[0]], dataLogInfeas[:,0,self.listLog[1]], dataLogInfeas[:,2,self.listLog[2]]), axis=1)

        #FinalIn=np.concatenate((FeasIn, InFeasIn, InFeasIn[tmpidx,:]), axis=0)
        #FinalOut=np.concatenate((dataFeasible, dataInFeasible, dataInFeasible[tmpidx,:]), axis=0)

        FinalIn=np.concatenate((FeasIn[tmpidx,:], InFeasIn), axis=0)
        FinalOut=np.concatenate((dataFeasible[tmpidx,:], dataInFeasible), axis=0)

        idx=list(range(FinalOut.shape[0]))
        random.shuffle(idx)


        #return dataInput, dataInstruct, graspFinalIn.reshape(-1,2,graspIn.shape[1]+2*self.numLogicals), graspFinalOut.reshape(-1,2,self.numLogicals), placeFinalIn.reshape(-1,3,placeIn.shape[1]+3*self.numLogicals), placeFinalOut.reshape(-1,3,self.numLogicals)
        return FinalIn[idx,:], FinalOut[idx,:]
        

    def step_decay(self,epoch):
        initial_lrate = self.lr
        drop = self.lr_drop
        epochs_drop = self.epoch_drop
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate


    def train(self, path_rai, model_dir, saveToFile=True):
        if not model_dir=='':
            Dappend="_new"
            Setappend=""
            if self.mode in [1,2,3,4]:
                model_dir=model_dir+"_final/"
                Setappend="_"+str(self.mode)
            elif self.mode in [5,6,7,8]:
                model_dir=model_dir+"_stack/"
                Setappend="_"+str(self.mode)
            else:
                NotImplementedError

        dataInput, dataOutput=self.reshapeInput(path_rai, model_dir)

        tbFeas = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/Feasible')

        if not(os.path.exists(path_rai+'/logs/'+self.timestamp+'/tmp')):
                os.makedirs(path_rai+'/logs/'+self.timestamp+'/tmp')

        saveFeas=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelFeasible.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)

        if saveToFile:
            if not(os.path.exists(path_rai+'/logs/'+self.timestamp)):
                os.makedirs(path_rai+'/logs/'+self.timestamp)

            with open(path_rai+'/logs/'+self.timestamp+'/paramsFeas.txt', 'w') as f:
                for item in self.params:
                    for i in item:
                        f.write(i+"\t".expandtabs(4))
                    f.write("\n")
            
            shutil.copyfile(path_rai+'/dataset'+Dappend+'/'+model_dir+'Sets.txt',path_rai+'/logs/'+self.timestamp+'/'+model_dir[:15]+'-SetsFeas'+Setappend+'.txt')

        final_losses=[["Feasible"]]

        print("------Train Network Feas. "+rai_policy.estimateT(self.epochs_feas)+"------")
        print("Training set size: "+str(dataInput.shape[0]))
        modelFeasibleHist=self.modelFeasible.fit(x={"goal": dataInput[:, :self.goallength], "state": dataInput[:, self.goallength:self.goallength+self.statelength], "action": dataInput[:, self.goallength+self.statelength:]},
                                                y={"feasActOut": dataOutput[:, 0:1], "feasSkeOut": dataOutput[:, 1:2], "feasAct2Out": dataOutput[:, 0:1]},
                                                batch_size=32, epochs=self.epochs_feas,
                                                shuffle=True, verbose=0, validation_split=self.val_split,
                                                callbacks=[tbFeas,keras.callbacks.LearningRateScheduler(self.step_decay), rai_policy.printEpoch(),saveFeas, keras.callbacks.TerminateOnNaN(), rai_policy.EarlyStopping(self.val_split, patience=30)])

        final_losses[0].append(modelFeasibleHist.history["loss"][-1])
        final_losses[0].append(modelFeasibleHist.history["val_loss"][-1])


        if saveToFile:
            self.modelFeasible.save(path_rai+'/logs/'+self.timestamp+'/modelFeasible.h5')

            with open(path_rai+'/logs/'+self.timestamp+'/SummaryLossFeas.txt', 'a+') as f:
                for floss in final_losses:
                    f.write(floss[0]+": loss "+str(floss[1])+" | val_loss "+str(floss[2])+"\n")
            
            shutil.rmtree(path_rai+'/logs/'+self.timestamp+'/tmp', ignore_errors=True)

        return modelFeasibleHist

#--------------------------------------------------------

class ClassifierFeasLSTM():
    def __init__(self): #no itialization -> either load or build
        pass

    def load_net(self,path_rai, model_dir=''):
        #print(path_rai)
        if not model_dir=='':
            if self.mode==1:
                model_dir=model_dir+'/'
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
            elif self.mode==13:
                model_dir=model_dir+'_mixed13/'
            elif self.mode==14:
                model_dir=model_dir+'_mixed14/'
            else:
                model_dir=model_dir+'/'

        if os.path.isfile(path_rai+'/logs/'+model_dir+'modelFeasibleLSTM.h5'):
            self.modelFeasible=tf.keras.models.load_model(path_rai+'/logs/'+model_dir+'modelFeasibleLSTM.h5')
        else:
            self.modelFeasible=None


    def build_net(self,
                goallength,
                statelength,
                numInstruct,
                epochs_feas=300,
                n_layers_feas=2,
                n_layers_feas2=0, 
                size_feas=200,
                timestamp="",
                lr=0.001,
                lr_drop=0.5,
                epoch_drop=100,
                clipnorm=1.,
                val_split=0.2,
                reg=0.001,
                reg0=0,
                mode=3,
                listLog=[],
                path_rai=dir_file
                ):
        
        self.goalEncoder=tf.keras.models.load_model(path_rai+'/logs/encoder/encoderGoal.h5', compile=False)

        self.goallength=goallength
        self.statelength=statelength
        self.numInstruct=numInstruct

        self.epochs_feas=epochs_feas
        self.hlayers_feas=n_layers_feas
        self.hlayers_feas2=n_layers_feas2
        self.size_feas=size_feas

        self.reg=reg
        self.reg0=reg0
        self.lr=lr
        self.lr_drop=lr_drop
        self.epoch_drop=epoch_drop
        self.val_split=val_split
        self.mode=mode

        self.listLog=listLog
        if self.listLog is []:
            for _ in range(3):
                self.listLog.append(list(range(statelength/3)))

        print(self.listLog)

        self.params=[]
        param=["learning rate"]
        param.append("init: "+str(self.lr))
        param.append("drop: "+str(self.lr_drop))
        param.append("epoch_drop: "+str(self.epoch_drop))
        param.append("clipnorm: "+str(clipnorm))
        param.append("val_split: "+str(self.val_split))
        param.append("dataset_mode: "+str(self.mode))
        self.params.append(param)
        
        if timestamp=="":
            now=datetime.datetime.now()
            self.timestamp=str(now.year)+str(now.month).zfill(2)+str(now.day).zfill(2)+"-"+str(now.hour).zfill(2)+str(now.minute).zfill(2)+str(now.second).zfill(2)+"_feasible"
            if mode==11:
                self.timestamp=self.timestamp+"11"
            elif mode==12:
                self.timestamp=self.timestamp+"12"
            elif mode==13:
                self.timestamp=self.timestamp+"13"
            elif mode==14:
                self.timestamp=self.timestamp+"14"
        else:
            self.timestamp=timestamp

        self.modelFeasible = self.build_Feasible()

        self.modelFeasible.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm), loss="binary_crossentropy") #decay=0.001/self.epochs_inst
        
    #------------------build subnetwork---------------

    def build_Feasible(self):
        # One Hot : Multi label classification
        param=["Instruct"]
        param.append("hlayers_feas: "+str(self.hlayers_feas2)+"_"+str(self.hlayers_feas))
        param.append("size_feas: "+str(self.size_feas))
        param.append("epochs_feas: "+str(self.epochs_feas))
        self.params.append(param)

        inputs = keras.Input(shape=(self.numInstruct+len(self.listLog[0])+len(self.listLog[1])+len(self.listLog[2]),), name='action')
        inputs0 = keras.Input(shape=(self.goallength,), name='goal')
        inputs1=self.goalEncoder(inputs0)
        inputs2 = keras.Input(shape=(None, self.statelength,), name='state')

        _, feasNet = build_FeasLSTM(inputs2, self.size_feas, "feasScope", "feasMid", n_layers=0, size=self.size_feas, reg=self.reg, output_activation=keras.activations.relu)
        feasActNet = build_FeasSub(keras.layers.concatenate([inputs1, feasNet, inputs]), 1, "feasActScope", "feasActOut", n_layers=self.hlayers_feas, size=self.size_feas, reg=self.reg0)
        feasSkeNet = build_FeasSub(keras.layers.concatenate([inputs1, feasNet, inputs]), 1, "feasSkeScope", "feasSkeOut", n_layers=self.hlayers_feas, size=self.size_feas, reg=self.reg0)
        feasAct2Net = build_FeasSub(keras.layers.concatenate([feasNet, inputs]), 1, "feasAct2Scope", "feasAct2Out", n_layers=self.hlayers_feas, size=self.size_feas, reg=self.reg0)

        #feasNet0 = build_FeasSub(keras.layers.concatenate([inputs1, feasNet, inputs]), 2, "feasScope", "feasOut", n_layers=self.hlayers_feas-1, size=self.size_feas)
        
        modelFeasible = keras.models.Model(inputs=[inputs0,inputs2, inputs], outputs=[feasActNet, feasSkeNet, feasAct2Net], name="feasNet")
        #modelFeasible = keras.models.Model(inputs=[inputs1,inputs2, inputs], outputs=feasNet0, name="feasNet")
        return modelFeasible


    #------------------Train and save Model---------------
    #----TODO train, adapt input etc
    def reshapeInput(self, path_rai, model_dir):

        #path_rai="/home/my/rai-python/v_MA"
        #model_dir="20200122-104545_mixed3/"

        if self.mode in [11,12,13,14] and True:
            Dappend="_new"
        else:
            Dappend=""

        if self.mode in [12,14]:
            NPappend="_feat"
        else:
            NPappend=""

        dataFeasible=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'Feasible.npy')

        dataInstruct=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'Instruction.npy') #None numinstr
        dataLogicals=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'Logicals.npy') #None 3 numlog


        if os.path.isfile(path_rai+'/dataset'+Dappend+'/'+model_dir+'InputPrev.npy'):
            dataInputPrev=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'InputPrev'+NPappend+'.npy') #None 4 inputsize

        else:
            dataInputPrev=np.concatenate((np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'InputPrev1'+NPappend+'.npy'),np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'InputPrev2'+NPappend+'.npy')), axis=0)
            if os.path.isfile(path_rai+'/dataset'+Dappend+'/'+model_dir+'InputPrev3'+NPappend+'.npy'):
                dataInputPrev=np.concatenate((dataInputPrev,np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'InputPrev3'+NPappend+'.npy')), axis=0)


        dataInstruct2=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'InFeasibleInstr.npy') #None numinstr
        dataLogicals2=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'InFeasibleLog.npy') #None 3 numlog

        dataInputPrev2=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'InFeasibleInputPrev'+NPappend+'.npy') #None 4 inputsize
        dataFeasible2=np.load(path_rai+'/dataset'+Dappend+'/'+model_dir+'InFeasible.npy')           #None2

        #----------------------------------------------------------------------------------
        #tmpidx=list((range(dataFeasible2.shape[0])))
        #tmpidx=list((range(dataFeasible.shape[0])))
        #random.shuffle(tmpidx)
        #tmpidx=tmpidx[:(dataFeasible.shape[0]-dataFeasible2.shape[0])]
        #tmpidx=tmpidx[:dataFeasible2.shape[0]]

        infeasible=np.where(~dataFeasible[:,0:1].any(axis=1))[0]
        feasible=np.where(dataFeasible[:,0:1].any(axis=1))[0]
        tmpidx=infeasible
        print(len(tmpidx), dataFeasible2.shape[0])
        if len(tmpidx)<dataFeasible2.shape[0]:
            random.shuffle(feasible)
            #tmpidx=tmpidx+feasible[:dataFeasible2.shape[0]-len(tmpidx)]
            tmpidx=np.concatenate((tmpidx,feasible[:int(1.5*dataFeasible2.shape[0])-len(tmpidx)]), axis=0)
        
        #FinalIn=np.concatenate((dataInputPrev[:,:,self.goallength:], dataInputPrev2[:,:,self.goallength:], dataInputPrev2[tmpidx,:,self.goallength:]), axis=0)
        #FinalOut=np.concatenate((dataFeasible, dataFeasible2, dataFeasible2[tmpidx,:]), axis=0)
        #FinalGoal=np.concatenate((dataInputPrev[:,0,:self.goallength], dataInputPrev2[:,0,:self.goallength], dataInputPrev2[tmpidx,0,:self.goallength]), axis=0)
        #FinalAct=np.concatenate((
        #            np.concatenate((dataInstruct, dataLogicals[:,1,self.listLog[0]], dataLogicals[:,0,self.listLog[1]], dataLogicals[:,2,self.listLog[2]]), axis=1),
        #            np.concatenate((dataInstruct2, dataLogicals2[:,1,self.listLog[0]], dataLogicals2[:,0,self.listLog[1]], dataLogicals2[:,2,self.listLog[2]]), axis=1),
        #            np.concatenate((dataInstruct2[tmpidx,:], dataLogicals2[tmpidx,:][:,1,self.listLog[0]], dataLogicals2[tmpidx,:][:,0,self.listLog[1]], dataLogicals2[tmpidx,:][:,2,self.listLog[2]]), axis=1)
        #        ),axis=0)

        FinalIn=np.concatenate((dataInputPrev[tmpidx,:,self.goallength:], dataInputPrev2[:,:,self.goallength:]), axis=0)
        FinalOut=np.concatenate((dataFeasible[tmpidx,:], dataFeasible2), axis=0)
        FinalGoal=np.concatenate((dataInputPrev[tmpidx,0,:self.goallength], dataInputPrev2[:,0,:self.goallength]), axis=0)
        FinalAct=np.concatenate((
                    np.concatenate((dataInstruct[tmpidx,:], dataLogicals[tmpidx,:][:,1,self.listLog[0]], dataLogicals[tmpidx,:][:,0,self.listLog[1]], dataLogicals[tmpidx,:][:,2,self.listLog[2]]), axis=1),
                    np.concatenate((dataInstruct2, dataLogicals2[:,1,self.listLog[0]], dataLogicals2[:,0,self.listLog[1]], dataLogicals2[:,2,self.listLog[2]]), axis=1),
                ),axis=0)


        idx=list(range(FinalIn.shape[0]))
        random.shuffle(idx)
        FinalIn=FinalIn[idx,:,:]
        FinalOut=FinalOut[idx,:]

        idxFinal=[[],[],[],[]]
        for i in range(FinalIn.shape[0]):
            for j in range(FinalIn.shape[1]):
                if np.any(FinalIn[i,j,:]) or j == 0:
                    if j==3:
                        idxFinal[3].append(i)
                else:
                    idxFinal[j-1].append(i)
                    break

        print(len(idxFinal[0]),len(idxFinal[1]),len(idxFinal[2]),len(idxFinal[3]))
        #print(FinalIn.shape)
        #print(FinalOut.shape)
        #print(FinalGoal.shape)
        #print(FinalAct.shape)

        FinalList = [[FinalIn[idxFinal[0], 0:1,:], FinalOut[idxFinal[0],:], FinalGoal[idxFinal[0],:], FinalAct[idxFinal[0],:]],
                     [FinalIn[idxFinal[1], 0:2,:], FinalOut[idxFinal[1],:], FinalGoal[idxFinal[1],:], FinalAct[idxFinal[1],:]],
                     [FinalIn[idxFinal[2], 0:3,:], FinalOut[idxFinal[2],:], FinalGoal[idxFinal[2],:], FinalAct[idxFinal[2],:]],
                     [FinalIn[idxFinal[3], 0:4,:], FinalOut[idxFinal[3],:], FinalGoal[idxFinal[3],:], FinalAct[idxFinal[3],:]],
                     [FinalIn, FinalOut, FinalGoal, FinalAct]
                     ]

        return FinalList
        

    def step_decay(self,epoch):
        initial_lrate = self.lr
        drop = self.lr_drop
        epochs_drop = self.epoch_drop
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate


    def train(self, path_rai, model_dir, num_batch_it, saveToFile=True):
        if not model_dir=='':
            Dappend="_new"
            Setappend=""
            if self.mode in [1,2,3,4]:
                model_dir=model_dir+"_final/"
                Setappend="_"+str(self.mode)
            elif self.mode in [5,6,7,8]:
                model_dir=model_dir+"_stack/"
                Setappend="_"+str(self.mode)
            else:
                NotImplementedError

        finalList=self.reshapeInput(path_rai, model_dir)

        tbFeas = TensorBoard(log_dir=path_rai+'/logs/'+self.timestamp+'/Feasible2')

        if not(os.path.exists(path_rai+'/logs/'+self.timestamp+'/tmp')):
                os.makedirs(path_rai+'/logs/'+self.timestamp+'/tmp')

        saveFeas=keras.callbacks.ModelCheckpoint(path_rai+'/logs/'+self.timestamp+'/tmp/modelFeasibleLSTM.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=50)

        if saveToFile:
            if not(os.path.exists(path_rai+'/logs/'+self.timestamp)):
                os.makedirs(path_rai+'/logs/'+self.timestamp)

            with open(path_rai+'/logs/'+self.timestamp+'/paramsFeas2.txt', 'w') as f:
                for item in self.params:
                    for i in item:
                        f.write(i+"\t".expandtabs(4))
                    f.write("\n")
            
            shutil.copyfile(path_rai+'/dataset'+Dappend+'/'+model_dir+'Sets.txt',path_rai+'/logs/'+self.timestamp+'/'+model_dir[:15]+'-SetsFeas2'+Setappend+'.txt')

        final_losses=[["Feasible"]]

        print("------Train Network Feas2. "+rai_policy.estimateT(self.epochs_feas)+"------")
        print("Training set size: "+str(finalList[4][0].shape[0]))
        i=4
        modelFeasibleHist=self.modelFeasible.fit(x={"goal":finalList[i][2], "state": finalList[i][0], "action":finalList[i][3]},
                                                y={"feasActOut": finalList[i][1][:,0:1], "feasSkeOut": finalList[i][1][:,1:2], "feasAct2Out": finalList[i][1][:,0:1]},
                                                #y=finalList[i][1],
                                                batch_size=32, epochs=self.epochs_feas,
                                                shuffle=True, verbose=0, validation_split=self.val_split,
                                                callbacks=[tbFeas,keras.callbacks.LearningRateScheduler(self.step_decay), rai_policy.printEpoch(),saveFeas, keras.callbacks.TerminateOnNaN(), rai_policy.EarlyStopping(self.val_split, patience=25)])


        final_losses[0].append(modelFeasibleHist.history["loss"][-1])
        final_losses[0].append(modelFeasibleHist.history["val_loss"][-1])


        if saveToFile:
            self.modelFeasible.save(path_rai+'/logs/'+self.timestamp+'/modelFeasibleLSTM.h5')

            with open(path_rai+'/logs/'+self.timestamp+'/SummaryLossFeas2.txt', 'a+') as f:
                for floss in final_losses:
                    f.write(floss[0]+": loss "+str(floss[1])+" | val_loss "+str(floss[2])+"\n")
            
            shutil.rmtree(path_rai+'/logs/'+self.timestamp+'/tmp', ignore_errors=True)

        return modelFeasibleHist

#--------------------------------------------------------

