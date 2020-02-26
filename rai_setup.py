import sys
import os

dir_file=os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir_file+'/../ry/')
from libry import *

import numpy as np
import rai_policy2 as NNencoder

import time

def splitCommandStep(stringNew, verbose=0):
    # Extracts every word from a command
    split_str = stringNew.split(" ")
    list_str = []

    for word in split_str:
        if(word is split_str[0]):
            list_str.append(word[1:])
        elif (word is split_str[-1]):
            list_str.append(word[:-1])
        else:
            list_str.append(word)
    if verbose>0:
        print(list_str)
    return list_str


def splitStringStep(stringNew, list_old=[], verbose=0):
    # Extracts every single high-level action from a skeleton
    split_str = stringNew.split(") (")
    list_str = list_old

    for word in split_str:
        if(word is split_str[0]):
            if(word is split_str[-1]):
                list_str.append(word)
            else:
                list_str.append(word+")")
        elif (word is split_str[-1]):
            list_str.append("("+word)
        else:
            list_str.append("("+word+")")
    if verbose>0:
        print(list_str)
    return list_str

def splitStringPath(stringNew, list_old=[], verbose=0):
    # Extracts every single partial skeleton from a skeleton
	split_str = stringNew.split(") (")
	list_str = list_old

	for word in split_str:
		if(word is split_str[0]):
			if(word is split_str[-1]):
				list_str.append(word)
			elif list_str==[]:
				list_str.append(word+")")
			else:
				list_str.append(list_str[-1]+" "+word+")")
		elif (word is split_str[-1]):
			list_str.append(list_str[-1]+" ("+word)
		else:
			list_str.append(list_str[-1]+" ("+word+")")
	if verbose>0:
		print(list_str)

	return list_str

def softmaxToOnehot(a):
    # Converts probability to one-hot encoding
	a[0, np.argmax(a, axis=1)]=1
	a[a<1]=0
	return a

def runLGP(lgp, bound, verbose=0, view=True): #BT.pose BT.seq BT.path
    # Runs LGP and returns komo for bound
	lgp.optBound(bound, True,view)
	if verbose>0:
		print("Bound", bound, "feasible: ", not lgp.isInfeasible())
	komo = lgp.getKOMOforBound(bound)
	return komo

def applyKomo(komo, logical, num=-1, verbose=0):
    # Extracts state(symbols only) and configuration (comlete)
	state = komo.get7dLogical(num, logical, len(logical))
	if verbose>0:
		print(state)
	config = komo.getConfiguration(num)
	return state, config

def rearrangeGoal(goal):
    # rearrange goal: (on obj table) -> (on table obj)
    # In order to adapt to cahnge of original cpp code
    goaltmp=goal.split(" ")
    if len(goaltmp)==3:
        return goaltmp[0]+" "+goaltmp[2][:-1]+" "+goaltmp[1]+")"
    else:
        return goal

#------------------------------------------------------------------------------------------------------------------------------
class RaiWorld():
    def __init__(self, path_rai, nenv, setup, goalString, verbose, NNmode="minimal", maxDepth=100, datasetMode=1, view=True):

        # Init some variables
        self.path_rai=path_rai
        self.verbose=verbose
        self.maxDepth=maxDepth
        self.setup=setup
        self.NNmode=NNmode
        self.nenv=nenv
        
        self.K=Config()
        self.feasweight=0

        if setup=="minimal":
            self.numLogical=3
            self.numGoal=2
            self.numGoalInstruct=2
            self.numActInstruct=2
            self.numObj=3

            # Load Configuration
            self.K.addFile(path_rai+'/rai-robotModels/pr2/pr2.g')
            self.K.addFile(path_rai+'/models/Test_setup_'+str(nenv).zfill(3)+'.g')

            self.logicalNames, self.logicalType , self.grNames, self.objNames, self.tabNames=self.preprocessLogicalState()
            if datasetMode in [1,2,3,4]:
                self.goallength=(self.numGoalInstruct + self.numObj+5)*self.numGoal

        # Extract original symbols (in case there are more than 3 objects)
        self.objNames_orig=self.objNames.copy()
        self.logicalNames_orig=self.logicalNames.copy()
        self.tabNames_orig=self.tabNames.copy()

        self.dataMode=datasetMode
        self.view=view
        if view:
            self.V = self.K.view()

        # In case relative pos (mode is 2 or 4)
        self.baseName=[self.K.getFrameNames()[1]]

        # Set goal
        self.redefine(goalString)


    def redefine(self,goalString, nenv=None):
        printInit=True
        if nenv is not None:
            # Reload configuration
            printInit=False
            del self.K
            del self.lgp
            time.sleep(1)
            self.nenv=nenv
            self.K=Config()
            self.K.addFile(self.path_rai+'/rai-robotModels/pr2/pr2.g')
            if self.setup=="minimal":
                self.K.addFile(self.path_rai+'/models/Test_setup_'+str(nenv).zfill(3)+'.g')
            if self.view:
                del self.V
                self.V = self.K.view()

        if self.setup=="minimal":
            # Load LGP
            self.lgp=self.K.lgp(self.path_rai+"/models/fol-pickAndPlace2.g", printInit)

        self.goalString_orig=goalString
        self.numGoal_orig=len(splitStringStep(self.goalString_orig, list_old=[],verbose=0))

        if not goalString=="":
            # Extract goal string for LGP: (on obj table) -> (on table obj)
            self.goalString=""
            goaltmp=splitStringStep(goalString, list_old=[])
            goalString=" ".join([rearrangeGoal(goal) for goal in goaltmp])
            self.realGoalString=goalString
            self.lgp.addTerminalRule(goalString)
            
            # Compute objective encoding
            self.goalState, _,_=self.preprocessGoalState(initState=True)
        else:
            self.realGoalString=goalString	
                
    #----------------------Preprocessing: Find relevant "logicals"-----------------------------------
    def preprocessLogicalState(self):
        logicalNames= self.K.getFrameNamesLogical()
        logicalType = self.K.getLogical(logicalNames) 

        logicalNamesFinal, logicalTypeFinal, grNames, objNames, tabNames = [], [], [], [], []

        # Get all symbols: have at least one logic type
        for ltype, lname in zip(logicalType, logicalNames):
            if not ltype == {}:
                logicalNamesFinal.append(lname)
                logicalTypeFinal.append(ltype)

        logicalTypesencoded=np.zeros((len(logicalNamesFinal), self.numLogical))

        listGr, listObj, listTab = [],[],[]
        for ltype, lname in zip(logicalTypeFinal, logicalNamesFinal):
            i=logicalNamesFinal.index(lname)
            logicalTypesencoded[i,:]= self.encodeLogicalMultiHot(ltype)
            # Assign all symbols to the sets of logic types (index only)
            if self.setup in ["minimal"]:
                if 'gripper' in ltype:
                    grNames.append(lname)
                    listGr.append(i)
                if 'object' in ltype:
                    objNames.append(lname)
                    listObj.append(i)
                if 'table' in ltype:
                    tabNames.append(lname)
                    listTab.append(i)

            else:
                NotImplementedError

        self.listLog=[listGr, listObj, listTab]

        if self.NNmode in ["minimal", "dataset", "mixed", "FFchain", "mixed2", "mixed0", "FFnew", "mixed3", "final", "mixed10"]:
            if self.setup=="minimal":
                return logicalNamesFinal, logicalTypesencoded , grNames, objNames, tabNames
            else:
                NotImplementedError
            
        elif self.NNmode in ["full", "chain", "3d"]:
            if self.setup=="minimal":
                return logicalNamesFinal, logicalTypesencoded , logicalNamesFinal, logicalNamesFinal, logicalNamesFinal
            else:
                NotImplementedError
        else:
            NotImplementedError

    def preprocessGoals(self):
        goalStep = splitStringStep(self.goalString_orig, list_old=[],verbose=0)
        goalStepReal = splitStringStep(self.realGoalString, list_old=[],verbose=0)
        folstate = self.lgp.nodeState()
        unfullfilled=[]

        for goal, real in zip(goalStep, goalStepReal):
            # Find unsatisfied goal formulations
            if not real in folstate[0]:
                unfullfilled.append(goal)
        return unfullfilled

    def preprocessGoalState(self, initState=False, cheatGoalState=False):
        # Determine unsatisfied objectives
        if initState:
            unfullfilled=splitStringStep(self.goalString_orig, list_old=[],verbose=0)
            if len(unfullfilled)==1:
                unfullfilled=[unfullfilled[0], unfullfilled[0]]
        else:
            unfullfilled=self.preprocessGoals()
        
        changed=False
        goalString_prev=self.goalString

        if self.setup=="minimal":
            # Determine if objective should be changed
            if len(unfullfilled)>2:
                changed=True

            elif len(unfullfilled)==1 and cheatGoalState:
                unfullfilled.append(unfullfilled[0])
                if not self.goalString==unfullfilled[0]+" "+unfullfilled[1]:
                    self.padInput()
                    changed=True
            elif self.numGoal_orig>2:
                changed=True
            else:
                changed=False
        else:
            NotImplementedError

        if not (len(unfullfilled)==1 and not cheatGoalState) and len(unfullfilled)>0:
            # If goal state should be adapted
            if self.goalString==unfullfilled[0]+" "+unfullfilled[1]:
                changed=False
            else:
                self.goalString=unfullfilled[0]+" "+unfullfilled[1]
        elif len(unfullfilled)==1 and not cheatGoalState:
            # If goal state should not be adapted
            unfullfilled=splitStringStep(goalString_prev, list_old=[],verbose=0)
            self.goalString=goalString_prev


        if len(self.objNames_orig)>self.numObj:
            # Select objects included in goal state only
            self.objNames=self.objNames_orig.copy()
            self.logicalNames=self.logicalNames_orig.copy()
            self.tabNames=self.tabNames_orig.copy()
            for obj in self.objNames_orig:
                if not obj in self.goalString:
                    self.objNames.remove(obj)
                    self.logicalNames.remove(obj)
                    self.tabNames.remove(obj)
                    if len(self.objNames)==self.numObj:
                        break
        
        if len(self.objNames)>self.numObj:
            # If too many objects: change objective to one goal formulation and select objects
            changed=True
            self.goalString=unfullfilled[0]+" "+unfullfilled[0]
            unfullfilled=[unfullfilled[0], unfullfilled[0]]

            self.objNames=self.objNames_orig.copy()
            self.logicalNames=self.logicalNames_orig.copy()
            self.tabNames=self.tabNames_orig.copy()
            for obj in self.objNames_orig:
                if not obj in self.goalString:
                    self.objNames.remove(obj)
                    self.logicalNames.remove(obj)
                    self.tabNames.remove(obj)
                    if len(self.objNames)==self.numObj:
                        break

        # Encode objective
        goalState=self.encodeGoal(unfullfilled[:2])

        return goalState, unfullfilled, changed


    #-----------------------Encoding: Logical Type, Goal State, State, Input, Action--------------------------
    def encodeLogicalMultiHot(self,ltype):
        # Multi-hot encoding of all logic types for symbol
        if self.setup in "minimal":
            multiHot= np.zeros((1,3), dtype=int)

            if 'gripper' in ltype:
                multiHot[(0,0)]=1
            if 'object' in ltype:
                multiHot[(0,1)]=1
            if 'table' in ltype:
                multiHot[(0,2)]=1
            #if 'pusher' in ltype:
            #	return 'pusher'
            #if 'wall' in ltype:
            #	return 'wall'

        else:
            NotImplementedError

        return multiHot

    def encodeGoal(self,goalStep):
        # Encode objective
        if self.setup in ["minimal"]:
            goalLen = self.numGoalInstruct+len(self.objNames)+len(self.tabNames)
            goalEn = np.zeros((1, self.numGoal*(goalLen)))

            for i,goalString in zip(range(len(goalStep)),goalStep):
                # For each goal specification
                split_str = goalString.split(" ")

                if split_str[0] == '(held':
                    # held object zeropadding
                    goalEn[(0,0+i*goalLen)]=1
                    split_str[1]=split_str[1][:-1]
                    goalEn[(0,self.numGoalInstruct+self.objNames.index(split_str[1])+i*goalLen)]=1
                elif split_str[0] == '(on':
                    # on object table
                    goalEn[(0,1+i*goalLen)]=1
                    goalEn[(0,self.numGoalInstruct+self.objNames.index(split_str[1])+i*goalLen)]=1
                    split_str[2]=split_str[2][:-1]
                    goalEn[(0,2+len(self.objNames)+self.tabNames.index(split_str[2])+i*goalLen)]=1

                else:
                    NotImplementedError

        else:
            NotImplementedError

        return goalEn

    def encodeState(self):
        # Encode state
        if self.NNmode=="final":
            # For data set: global coordinates and relative to base
            return [self.K.get7dLogical(self.logicalNames, len(self.logicalNames))[:,0:3], self.encodeFeatures2()]
        elif self.dataMode in [1,3]:
            # Global coordinates
            return self.K.get7dLogical(self.logicalNames, len(self.logicalNames))[:,0:3]
        elif self.dataMode in [2,4]:
            # Coordinates relative to base
            return self.encodeFeatures2()
        else:
            NotImplementedError

    def encodeFeatures2(self):
        # Get coordinates relative to base
        return self.K.getfeatures2DLogical(self.logicalNames, len(self.logicalNames), self.baseName)

    def encodeInput(self, envState, goalState=None):
        # Input encoding
        if goalState is None:
            goalState=self.goalState
        if self.NNmode=="final":
            # For data set: global coordinates and relative to base
            return [np.concatenate((goalState,np.reshape(envState[0], (1,-1))), axis=1), np.concatenate((goalState,np.reshape(envState[1], (1,-1))), axis=1)]
        else:
            # Concatenate objective encoding and state encoding
            return np.concatenate((goalState,np.reshape(envState, (1,-1))), axis=1)

    def encodeAction(self, commandString):
        # Output encoding for high-level action
        if self.setup=="minimal":
            instructionEn = np.zeros((1,self.numActInstruct))
            split_str = commandString.split(" ")

            if split_str[0] == '(grasp':
                # instruction
                instructionEn[(0,0)] = 1
                # gripper
                logEn = np.zeros((1, len(self.grNames)+len(self.objNames)))
                logEn[(0,self.grNames.index(split_str[1]))]=1
                # object
                split_str[2]=split_str[2][:-1]
                logEn[(0,len(self.grNames)+self.objNames.index(split_str[2]))]=1

            elif split_str[0] == '(place':
                # instruction
                instructionEn[(0,1)] = 1
                # gripper
                logEn = np.zeros((1, len(self.grNames)+len(self.objNames)+len(self.tabNames)))
                logEn[(0,self.grNames.index(split_str[1]))]=1
                # object
                split_str[2]=split_str[2]
                logEn[(0,len(self.grNames)+self.objNames.index(split_str[2]))]=1
                # table
                split_str[3]=split_str[3][:-1]
                logEn[(0,len(self.grNames)+len(self.objNames)+self.tabNames.index(split_str[3]))]=1

            else:
                logEn = np.zeros((1, 1))
                NotImplementedError

            return instructionEn, logEn
        else:
            NotImplementedError

    def checkPlaceSame(self, commandString):
        # Checks if object should be placed on itself
        # Probably not required anymore doe to new fol rule?
        if self.setup=="minimal":
            split_str = commandString.split(" ")

            if split_str[0] == '(grasp':
                return False

            elif split_str[0] == '(place':

                if split_str[2]==split_str[3][:-1]:
                    return True
                else:
                    return False
            else:
                NotImplementedError

        else:
            NotImplementedError

    #-----------------------------Decoding: Action ----------------------------------------------------------
    
    def decodeAction(self,instructionEn, logEn):
        if instructionEn[0,0] == 1 and instructionEn[0,1] == 0:
            # Decode grasp
            commandString='(grasp ' + self.grNames[logEn[0,0:len(self.grNames)].argmax()] + ' ' + self.objNames[logEn[0,len(self.grNames):].argmax()] +')'
            
        elif instructionEn[0,1] == 1 and instructionEn[0,0] == 0:
            # Decode place
            commandString='(place ' + self.grNames[logEn[0,0:len(self.grNames)].argmax()] + ' ' + self.objNames[logEn[0,len(self.grNames):len(self.grNames)+len(self.objNames)].argmax()]+ ' ' + self.tabNames[logEn[0,len(self.grNames)+len(self.objNames):].argmax()]  +')'

        else:
            NotImplementedError

        return commandString

    def decodeAction1(self,actionEn):
        instructionEn=actionEn[:,0:self.numActInstruct]
        logEn=actionEn[:,self.numActInstruct:]

        return self.decodeAction(instructionEn, logEn)

    #--------------------------Policy: Currently Feed Forward NN-------------------------------------------------------

    def saveFit(self, model_dir,epochs_inst, n_layers_inst, n_size_inst, epochs_grasp, n_layers_grasp, n_size_grasp, epochs_place, n_layers_place, n_size_place,
                lr, lr_drop, epoch_drop,clipnorm, val_split, reg, reg0, batch_size, n_layers_inst2=0):
        
        # Trains hierarchical policy

        if self.NNmode=="minimal":
            # Implementation 1
            input_size = (self.numGoalInstruct + len(self.listLog[1])+len(self.listLog[2]))*self.numGoal + len(self.logicalNames)*3

            self.rai_net = NNencoder.FeedForwardNN()
            self.rai_net.build_net(input_size,self.numActInstruct,len(self.grNames),len(self.objNames),len(self.tabNames),
                                epochs_inst=epochs_inst, n_layers_inst=n_layers_inst, size_inst=n_size_inst,
                                epochs_grasp=epochs_grasp, n_layers_grasp=n_layers_grasp, size_grasp=n_size_grasp,
                                epochs_place=epochs_place, n_layers_place=n_layers_place, size_place=n_size_place,
                                lr=lr, lr_drop=lr_drop, epoch_drop=epoch_drop, val_split=val_split, mode=self.dataMode,
                                listLog=self.listLog, reg0=reg0, batch_size=batch_size
                                )

            modelInstructHist, modelGraspHist, modelPlaceHist=self.rai_net.train(self.path_rai, model_dir)
            self.model_dir=self.rai_net.timestamp

            return modelInstructHist, modelGraspHist, modelPlaceHist

        elif self.NNmode in ["mixed10"]:
            # Implementation 3
            input_size = (self.numGoalInstruct + len(self.objNames)+len(self.tabNames))*self.numGoal + len(self.logicalNames)*3

            self.prevInput = np.zeros((5,input_size))
            self.input_size=input_size
            self.step=0
            

            self.rai_net=NNencoder.ClassifierMixed()

            self.rai_net.build_net(input_size,self.numActInstruct,len(self.logicalNames),
                                epochs_inst=epochs_inst, n_layers_inst=n_layers_inst, size_inst=n_size_inst,
                                epochs_grasp=epochs_grasp, n_layers_grasp=n_layers_grasp, size_grasp=n_size_grasp,
                                epochs_place=epochs_place, n_layers_place=n_layers_place, size_place=n_size_place,
                                lr=lr, lr_drop=lr_drop, epoch_drop=epoch_drop, clipnorm=clipnorm, val_split=val_split,
                                reg=reg, listLog=self.listLog, mode=self.dataMode,
                                n_layers_inst2=n_layers_inst2, reg0=reg0, batch_size=batch_size
                                )
            modelInstructHist, modelGraspObjHist, modelGraspGrpHist, modelPlaceObjHist, modelPlaceGrpHist=self.rai_net.train(self.path_rai, model_dir)
            self.model_dir=self.rai_net.timestamp.split("_")[0]

        elif self.NNmode=="FFnew":
            # Implementation 2
            input_size = (self.numGoalInstruct + len(self.listLog[1])+len(self.listLog[2]))*self.numGoal + len(self.logicalNames)*3
            self.rai_net=NNencoder.ClassifierChainNew()

            self.rai_net.build_net(input_size,self.numActInstruct,len(self.logicalNames),
                                epochs_inst=epochs_inst, n_layers_inst=n_layers_inst, size_inst=n_size_inst,
                                epochs_grasp=epochs_grasp, n_layers_grasp=n_layers_grasp, size_grasp=n_size_grasp,
                                epochs_place=epochs_place, n_layers_place=n_layers_place, size_place=n_size_place,
                                lr=lr, lr_drop=lr_drop, epoch_drop=epoch_drop, clipnorm=clipnorm, val_split=val_split,
                                reg0=reg0, listLog=self.listLog, mode=self.dataMode, batch_size=batch_size
                                )
            modelInstructHist, modelGraspObjHist, modelGraspGrpHist, modelPlaceObjHist, modelPlaceGrpHist=self.rai_net.train(self.path_rai, model_dir)
            self.model_dir = self.rai_net.timestamp.split("_")[0]

        else:
            NotImplementedError


    def loadFit(self, model_dir=''):
        # Load Policy
        if self.NNmode=="minimal":
            # Implementation 1
            self.rai_net = NNencoder.FeedForwardNN()
            self.rai_net.mode=self.dataMode
       
        elif self.NNmode in ["mixed10"]:
            # Implementation 3
            self.rai_net=NNencoder.ClassifierMixed()
            self.rai_net.mode=self.dataMode
            self.input_size = (self.numGoalInstruct + len(self.objNames)+len(self.tabNames))*self.numGoal + len(self.logicalNames)*3
            self.prevInput = np.zeros((5,self.input_size))
            self.step=0

        elif self.NNmode=="FFnew":
            # Implementaion 2
            self.rai_net=NNencoder.ClassifierChainNew()
            self.rai_net.mode=self.dataMode

        else:
            NotImplementedError
        self.rai_net.load_net(self.path_rai, model_dir)
        self.model_dir=model_dir


    def resetFit(self, cheatGoalState=False, goal=""):
        # Reset goal (encoding)
        if self.NNmode in ["mixed10"]:
            self.prevInput = np.zeros((5,self.input_size))
            self.step=0
        self.goalString=goal
        self.goalString_orig=goal
        self.goalState, _,_=self.preprocessGoalState(initState=True)

    #--------------Action Prediction: NN Softmax output to one hot, most probable action of all possible decisons-------------
    def padInput(self):
        # Adapt prevInput for lstm to improve performance of cheat_goalstate mode
        if self.NNmode in ["mixed10"] and self.step==1:
            self.prevInput = np.zeros((5,self.input_size))
            self.step=0

    def processPrediction(self,inputState, oneHot=True):
        # Predict next high-level action
        if self.setup=="minimal":
            # Extract objective and state from input
            goalState=inputState[:,:self.goallength]
            envState=inputState[:,self.goallength:]
            if self.NNmode=="minimal":
                # Implementation 1

                # Get instruction
                instrPred = self.rai_net.modelInstruct.predict({"goal": goalState, "state": envState})

                if oneHot:
                    instrPred = softmaxToOnehot(instrPred)
                else:
                    instrPred=instrPred.reshape((1,-1))

                if instrPred[0,0]==1 or np.argmax(instrPred)==0: #grasp
                    # Get gripper and object
                    graspPred=self.rai_net.modelGrasp.predict({"goal": goalState, "state": envState})

                    if oneHot:
                        graspPred=np.concatenate((softmaxToOnehot(graspPred[0]),softmaxToOnehot(graspPred[1])), axis=1)
                    else:
                        graspPred=np.concatenate((graspPred[0],graspPred[1]), axis=1)

                    return np.concatenate((instrPred,graspPred), axis=1)

                elif instrPred[0,1]==1 or np.argmax(instrPred)==1: #place
                    # Get gripper object and table
                    placePred=self.rai_net.modelPlace.predict({"goal": goalState, "state": envState})
                    if oneHot:
                        placePred=np.concatenate((softmaxToOnehot(placePred[0]),softmaxToOnehot(placePred[1]),softmaxToOnehot(placePred[2])), axis=1)
                    else:
                        placePred=np.concatenate((placePred[0],placePred[1],placePred[2]), axis=1)
                    
                    return np.concatenate((instrPred,placePred), axis=1)

                else:
                    NotImplementedError


            elif self.NNmode in ["mixed10"]:
                # Implementation 3
                # Add input to input sequence
                if self.step <5:
                    self.prevInput[self.step, : self.input_size] = inputState
                else:
                    self.prevInput = np.concatenate((self.prevInput[1:, : ], inputState), axis=0)
                self.step+=1

                # Get Instruction
                instrPred = self.rai_net.modelInstruct.predict({"goal": goalState.reshape((1,1,-1)),
                                                                "state":self.prevInput[0:4,self.goallength:].reshape((1,4, -1))})

                if oneHot:
                    instrPred = softmaxToOnehot(instrPred)
                else:
                    instrPred=instrPred.reshape((1,-1))

                if instrPred[0,0]==1 or np.argmax(instrPred)==0: #grasp
                    graspPred=[None, None]
                    # Get object
                    graspPred[1]=self.rai_net.modelGraspObj.predict({"goal": goalState, "state1": envState})

                    if oneHot:
                        graspPred[1]=softmaxToOnehot(graspPred[1])
                    
                    # Get gripper
                    graspPred[0]=self.rai_net.modelGraspGrp.predict({"goal": goalState, "state2":np.concatenate((envState, graspPred[1]), axis=1)})

                    if oneHot:
                        graspPred[0]=softmaxToOnehot(graspPred[0])
                    
                    graspPred=np.concatenate((graspPred[0].reshape((1,-1)),graspPred[1].reshape((1,-1))), axis=1)

                    return np.concatenate((instrPred,graspPred), axis=1)
                    
                elif instrPred[0,1]==1 or np.argmax(instrPred)==1: #place
                    placePred=[None, None, None]
                    # Get object
                    placePred[1]=self.rai_net.modelPlaceObj.predict({"goal": goalState, "state1": envState})

                    if oneHot:
                        placePred[1] = softmaxToOnehot(placePred[1])

                    # Get gripper and table
                    [placePred[0], placePred[2]]=self.rai_net.modelPlaceGrpTab.predict({"goal": goalState, "state2":np.concatenate((envState, placePred[1]), axis=1)})

                    if oneHot:
                        placePred[0] = softmaxToOnehot(placePred[0])
                        placePred[2] = softmaxToOnehot(placePred[2])
                        
                    placePred=np.concatenate((placePred[0].reshape((1,-1)),placePred[1].reshape((1,-1)),placePred[2].reshape((1,-1))), axis=1)
                        
                    return np.concatenate((instrPred,placePred), axis=1)

                else:
                    NotImplementedError

            elif self.NNmode in ["FFnew"]:
                # Implementation 2
                # Get instruction
                instrPred = self.rai_net.modelInstruct.predict({"goal": goalState, "state": envState})

                if oneHot:
                    instrPred = softmaxToOnehot(instrPred)
                else:
                    instrPred=instrPred.reshape((1,-1))

                if instrPred[0,0]==1 or np.argmax(instrPred)==0: #grasp
                    graspPred=[None, None]
                    # Get object
                    graspPred[1]=self.rai_net.modelGraspObj.predict({"goal": goalState, "state1": envState})

                    if oneHot:
                        graspPred[1]=softmaxToOnehot(graspPred[1])
                    
                    # Get gripper
                    graspPred[0]=self.rai_net.modelGraspGrp.predict({"goal": goalState, "state2":np.concatenate((envState, graspPred[1]), axis=1)})

                    if oneHot:
                        graspPred[0]=softmaxToOnehot(graspPred[0])
                    
                    graspPred=np.concatenate((graspPred[0].reshape((1,-1)),graspPred[1].reshape((1,-1))), axis=1)

                    return np.concatenate((instrPred,graspPred), axis=1)

                elif instrPred[0,1]==1 or np.argmax(instrPred)==1: #place
                    placePred=[None, None, None]
                    # Get object
                    placePred[1]=self.rai_net.modelPlaceObj.predict({"goal": goalState, "state1": envState})

                    if oneHot:
                        placePred[1] = softmaxToOnehot(placePred[1])

                    # Get gripper and table
                    [placePred[0], placePred[2]]=self.rai_net.modelPlaceGrpTab.predict({"goal": goalState, "state2":np.concatenate((envState, placePred[1]), axis=1)})
                    
                    if oneHot:
                        placePred[0] = softmaxToOnehot(placePred[0])
                        placePred[2] = softmaxToOnehot(placePred[2])
                        
                    placePred=np.concatenate((placePred[0].reshape((1,-1)),placePred[1].reshape((1,-1)),placePred[2].reshape((1,-1))), axis=1)
                        
                    return np.concatenate((instrPred,placePred), axis=1)

                else:
                    NotImplementedError

            else:
                NotImplementedError #NNmode
           
        else:
            NotImplementedError #setup

    def evalPredictions(self, inputState, infeasible=[""], maxdepth=[""], prevSke="", depth=0, tries=0):
        # Evaluate heuristic for all possible high-level actions for this node
        decisions=self.lgp.getDecisions()

        if self.setup=="minimal":
            # Extract objective and state from input
            goalState=inputState[:,:self.goallength]
            envState=inputState[:,self.goallength:]

            if self.NNmode=="minimal":
                # Get probability for instruction and symbols
                instrPred = self.rai_net.modelInstruct.predict({"goal": goalState, "state": envState})
                graspPred = self.rai_net.modelGrasp.predict({"goal": goalState, "state": envState})
                placePred = self.rai_net.modelPlace.predict({"goal": goalState, "state": envState})

            elif self.NNmode in ["mixed10"]:
                # Get probability for instruction and object
                instrPred = self.rai_net.modelInstruct.predict({"goal": goalState.reshape((1,1,-1)),
                                                                    "state":self.prevInput[0:4,self.goallength:].reshape((1,4, -1))})

                graspPred = [None, None]
                placePred = [None, None, None]

                graspPred[1]=self.rai_net.modelGraspObj.predict({"goal": goalState, "state1": envState})
                placePred[1]=self.rai_net.modelPlaceObj.predict({"goal": goalState, "state1": envState})

            
            elif self.NNmode in ["FFnew"]:
                graspPred = [None, None]
                placePred = [None, None, None]

                # Get probability for instruction and object
                instrPred = self.rai_net.modelInstruct.predict({"goal": goalState, "state": envState})
                graspPred[1]=self.rai_net.modelGraspObj.predict({"goal": goalState, "state1": envState})
                placePred[1]=self.rai_net.modelPlaceObj.predict({"goal": goalState, "state1": envState})


            else:
                NotImplementedError
            
            probBest=0
            actBest=""
            for decision, i in zip(decisions, range(len(decisions))):
                validAct=True
                for log in splitCommandStep(decision, verbose=0):
                    # If more than 3 objects: check if only input objects are included in high-level action
                    if log in self.objNames_orig and not log in self.logicalNames:
                        validAct=False
                        continue
                
                if not validAct:
                    continue

                # Get encoding of high-level action
                instructionEn, logEn = self.encodeAction(decision)

                if np.argmax(instructionEn, axis=1) == 0: #grasp gripper obj
                    # Get encoding of logic type object for high-level action
                    graspPrev = logEn[:,len(self.grNames):len(self.grNames)+len(self.objNames)]
                    if self.NNmode in ["mixed10", "FFnew"]:
                        # Get probabilty of gripper
                        graspPred[0]=self.rai_net.modelGraspGrp.predict({"goal": goalState, "state2":np.concatenate((envState, graspPrev), axis=1)})
                    
                    # Probabilty of object for consitional prob
                    if self.NNmode in ["mixed10", "FFnew"]:
                        tmpProb=np.inner(np.concatenate((np.zeros((1,len(self.grNames))),graspPred[1]), axis=1), logEn)
                    else:
                        tmpProb=1

                    # Calculate probabilty: weighting
                    prob = np.inner(instrPred, instructionEn) + 2*np.inner(np.concatenate((0.1*tmpProb*graspPred[0],0.9*graspPred[1]), axis=1), logEn)
                    prob[0,0]=prob[0,0]/3

                elif np.argmax(instructionEn, axis=1) == 1 :#place gripper obj table
                    # Get encoding of logic type object for high-level action
                    placePrev = logEn[:,len(self.grNames):len(self.grNames)+len(self.objNames)]
                    if self.checkPlaceSame(decision):
                        continue

                    if self.NNmode in ["FFnew","mixed10"]:
                        # Get probabilty of gripper and table
                        [placePred[0], placePred[2]]=self.rai_net.modelPlaceGrpTab.predict({"goal": goalState, "state2":np.concatenate((envState, placePrev), axis=1)})

                    # Probabilty of object for consitional prob
                    if self.NNmode in ["mixed10", "FFnew"]:
                        tmpProb=np.inner(np.concatenate((np.zeros((1,len(self.grNames))),placePred[1],np.zeros((1,len(self.tabNames)))), axis=1), logEn)
                    else:
                        tmpProb=1

                    # Calculate probabilty: weighting
                    prob = np.inner(instrPred, instructionEn) + 2*np.inner(np.concatenate((0.1*tmpProb*placePred[0],0.45*placePred[1],tmpProb*0.45*placePred[2]), axis=1), logEn)
                    prob[0,0]=prob[0,0]/3

                # Reduce heuristic for high-level actions that lead to skeletons labeled infeasible
                if decision in infeasible:
                    # infeasible
                    if tries<4:
                        penalty=(1.1-0.2*depth*(infeasible+maxdepth).count(decision))
                    else:
                        penalty=(1-0.3*depth*(infeasible+maxdepth).count(decision)**2)
                elif decision in maxdepth:
                    # maximum depth reached: greater reduction
                    if tries<4:
                        penalty=(1.1-0.3*(depth+1)*(infeasible+maxdepth).count(decision))
                        #penalty=(1.0-0.2*(depth+1)*(infeasible+maxdepth).count(decision))
                    else:
                        penalty=(1-0.2*(depth+1)*(infeasible+maxdepth).count(decision)**2)
                else:
                    penalty=1

                if penalty < 0.1:
                    penalty = 0.1/(depth*(infeasible+maxdepth).count(decision))
                
                prob=prob*penalty

                #print(decision, prob[0,0])
                
                if prob[0,0] > probBest:
                    # Select more probable high-level action
                    actBest= decisions[i]
                    probBest=prob[0,0]

            return actBest, probBest
        else:
            NotImplementedError