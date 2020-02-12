import sys
import os

dir_file=os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir_file+'/../ry/')
from libry import *

import numpy as np
import rai_policy2 as NNencoder

import time

def splitCommandStep(stringNew, verbose=0):
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
	a[0, np.argmax(a, axis=1)]=1
	a[a<1]=0
	return a

def runLGP(lgp, bound, verbose=0, view=True): #BT.pose BT.seq BT.path
	lgp.optBound(bound, True,view)
	if verbose>0:
		print("Bound", bound, "feasible: ", not lgp.isInfeasible())
	komo = lgp.getKOMOforBound(bound)
	return komo

# Extracts state(of logical objects only) and configuration (comlete)
def applyKomo(komo, logical, num=-1, verbose=0):
	state = komo.get7dLogical(num, logical, len(logical))
	if verbose>0:
		print(state)
	config = komo.getConfiguration(num)
	return state, config

def rearrangeGoal(goal):
    goaltmp=goal.split(" ")
    if len(goaltmp)==3:
        return goaltmp[0]+" "+goaltmp[2][:-1]+" "+goaltmp[1]+")"
    else:
        return goal

def rearrangeGoalTable(goal):
    goaltmp=goal.split(" ")
    if len(goaltmp)==3 and goaltmp[2][:-2]=="table":
        return goaltmp[0]+" "+goaltmp[2][:-1]+" "+goaltmp[1]+")"
    else:
        return goal


#------------------------------------------------------------------------------------------------------------------------------
class RaiWorld():
    def __init__(self, path_rai, nenv, setup, goalString, verbose, NNmode="minimal", maxDepth=100, datasetMode=1, view=True):

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

            self.K.addFile(path_rai+'/rai-robotModels/pr2/pr2.g')

            self.K.addFile(path_rai+'/models/Test_setup_'+str(nenv).zfill(3)+'.g')

            self.logicalNames, self.logicalType , self.grNames, self.objNames, self.tabNames=self.preprocessLogicalState()
            if datasetMode in [1,2,3,4]:
                self.goallength=(self.numGoalInstruct + self.numObj+5)*self.numGoal

        self.objNames_orig=self.objNames.copy()
        self.logicalNames_orig=self.logicalNames.copy()
        self.tabNames_orig=self.tabNames.copy()

        self.dataMode=datasetMode
        self.view=view
        if view:
            self.V = self.K.view()

        self.baseName=[self.K.getFrameNames()[1]]
        self.redefine(goalString)


    def redefine(self,goalString, nenv=None):
        printInit=True
        if nenv is not None:
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
            self.lgp=self.K.lgp(self.path_rai+"/models/fol-pickAndPlace.g", printInit)


        self.goalString_orig=goalString
        self.numGoal_orig=len(splitStringStep(self.goalString_orig, list_old=[],verbose=0))
        if not goalString=="":
            self.goalString=""
            #goalStep = splitStringStep(goalString, list_old=[],verbose=verbose)
            #goaltmp=splitStringStep(goalString, list_old=[])
            #goalString=" ".join([rearrangeGoal(goal) for goal in goaltmp])

            self.lgp.addTerminalRule(goalString)
            print(goalString)
            self.goalState, _,_=self.preprocessGoalState(initState=True)

        #self.lgp.walkToNode("(grasp pr2R red) (grasp pr2L green)",0)
        #komo = runLGP(self.lgp, BT.path, verbose=0, view=True)
        #input(self.lgp.nodeInfo())		
                
    #----------------------Preprocessing: Find relevant "logicals"-----------------------------------
    def preprocessLogicalState(self):
        logicalNames= self.K.getFrameNamesLogical()
        logicalType = self.K.getLogical(logicalNames) 

        logicalNamesFinal, logicalTypeFinal, grNames, objNames, tabNames = [], [], [], [], []

        for ltype, lname in zip(logicalType, logicalNames):
            if not ltype == {}:
                logicalNamesFinal.append(lname)
                logicalTypeFinal.append(ltype)

        logicalTypesencoded=np.zeros((len(logicalNamesFinal), self.numLogical))

        i=0
        listGr, listObj, listTab = [],[],[]
        for ltype, lname in zip(logicalTypeFinal, logicalNamesFinal):
            logicalTypesencoded[logicalNamesFinal.index(lname),:]= self.encodeLogicalMultiHot(ltype)
            if self.setup in ["minimal","pusher"]:
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
            i=i+1

        self.listLog=[listGr, listObj, listTab]

        if self.NNmode in ["minimal", "dataset", "mixed", "FFchain", "mixed2", "mixed0", "FFnew", "mixed3", "final", "mixed10"]:
            if self.setup=="minimal":
                return logicalNamesFinal, logicalTypesencoded , grNames, objNames, tabNames
            
        elif self.NNmode in ["full", "chain", "3d"]:
            if self.setup=="minimal":
                return logicalNamesFinal, logicalTypesencoded , logicalNamesFinal, logicalNamesFinal, logicalNamesFinal
            
        else:
            NotImplementedError

    def preprocessGoals(self):
        goalStep = splitStringStep(self.goalString_orig, list_old=[],verbose=0)
        folstate = self.lgp.nodeState()
        unfullfilled=[]
        for goal in goalStep:
            goaltmp=goal.split(" ")
            if goaltmp[-1][:-2]=="table":
                goal2=goaltmp[0]+" "+goaltmp[2][:-1]+" "+goaltmp[1]+")"
            else:
                goal2=goal
            if not (goal in folstate[0] or goal2 in folstate[0]):
                unfullfilled.append(goal)
        return unfullfilled

    def preprocessGoalState(self, initState=False, cheatGoalState=False):
        if initState:
            unfullfilled=splitStringStep(self.goalString_orig, list_old=[],verbose=0)
        else:
            unfullfilled=self.preprocessGoals()
        #input(unfullfilled)

        changed=False
        #goalString_prev=self.goalString
        #print(goalString_prev)

        if self.setup=="minimal":
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
            if self.goalString==unfullfilled[0]+" "+unfullfilled[1]:
                changed=False
            else:
                self.goalString=unfullfilled[0]+" "+unfullfilled[1]
            #print(self.goalString)

        if len(self.objNames_orig)>self.numObj:
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
            changed=True
            #self.goalString=goalString_prev
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

        #print(self.goalString)
        #print(self.tabNames)
        goalState=self.encodeGoal(unfullfilled[:2])

        return goalState, unfullfilled, changed


    #-----------------------Encoding: Logical Type, Goal State, State, Input, Action--------------------------
    def encodeLogicalMultiHot(self,ltype):

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

        if self.setup in ["minimal"]:
            goalLen = self.numGoalInstruct+len(self.objNames)+len(self.tabNames)
            goalEn = np.zeros((1, self.numGoal*(goalLen)))

            for i,goalString in zip(range(len(goalStep)),goalStep):
                split_str = goalString.split(" ")

                if split_str[0] == '(held':
                    goalEn[(0,0+i*goalLen)]=1
                    split_str[1]=split_str[1][:-1]
                    goalEn[(0,self.numGoalInstruct+self.objNames.index(split_str[1])+i*goalLen)]=1
                elif split_str[0] == '(on':
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
        if self.NNmode=="final":
            return [self.K.get7dLogical(self.logicalNames, len(self.logicalNames))[:,0:3], self.encodeFeatures2()]
        elif self.dataMode in [1,3]:
            return self.K.get7dLogical(self.logicalNames, len(self.logicalNames))[:,0:3]
        elif self.dataMode in [2,4]:
            return self.encodeFeatures2()
        else:
            NotImplementedError

    def encodeFeatures2(self):
        return self.K.getfeatures2DLogical(self.logicalNames, len(self.logicalNames), self.baseName)

    def encodeInput(self, envState, goalState=None):
        if goalState is None:
            goalState=self.goalState
        if self.NNmode=="final":
            return [np.concatenate((goalState,np.reshape(envState[0], (1,-1))), axis=1), np.concatenate((goalState,np.reshape(envState[1], (1,-1))), axis=1)]
        else:
            return np.concatenate((goalState,np.reshape(envState, (1,-1))), axis=1)

    def encodeAction(self, commandString):
        if self.setup=="minimal":
            instructionEn = np.zeros((1,self.numActInstruct))
            split_str = commandString.split(" ")

            if split_str[0] == '(grasp':
                instructionEn[(0,0)] = 1
                logEn = np.zeros((1, len(self.grNames)+len(self.objNames)))
                logEn[(0,self.grNames.index(split_str[1]))]=1

                split_str[2]=split_str[2][:-1]
                logEn[(0,len(self.grNames)+self.objNames.index(split_str[2]))]=1

            elif split_str[0] == '(place':
                instructionEn[(0,1)] = 1
                logEn = np.zeros((1, len(self.grNames)+len(self.objNames)+len(self.tabNames)))
                logEn[(0,self.grNames.index(split_str[1]))]=1

                split_str[2]=split_str[2]
                logEn[(0,len(self.grNames)+self.objNames.index(split_str[2]))]=1

                split_str[3]=split_str[3][:-1]
                logEn[(0,len(self.grNames)+len(self.objNames)+self.tabNames.index(split_str[3]))]=1

            else:
                logEn = np.zeros((1, 1))
                NotImplementedError

            return instructionEn, logEn
        else:
            NotImplementedError

    def checkPlaceSame(self, commandString):
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

    def encodeList(self, output, outputList):
        if self.NNmode in ["FFnew"]:
            newoutput = np.zeros((1,len(self.logicalNames)))
            for i in range(len(outputList)):
                newoutput[outputList[i]]=output[i]
            
            return newoutput
        
        else:
            NotImplementedError

    #-----------------------------Decoding: Action ----------------------------------------------------------
    
    def decodeAction(self,instructionEn, logEn):
        if instructionEn[0,0] == 1 and instructionEn[0,1] == 0:
            commandString='(grasp ' + self.grNames[logEn[0,0:len(self.grNames)].argmax()] + ' ' + self.objNames[logEn[0,len(self.grNames):].argmax()] +')'
            
        elif instructionEn[0,1] == 1 and instructionEn[0,0] == 0:
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
                lr, lr_drop, epoch_drop,clipnorm, val_split, reg, reg0, num_batch_it, n_layers_inst2=0):

        if self.NNmode=="minimal":

            input_size = (self.numGoalInstruct + len(self.listLog[1])+len(self.listLog[2]))*self.numGoal + len(self.logicalNames)*3

            self.rai_net = NNencoder.FeedForwardNN()
            self.rai_net.build_net(input_size,self.numActInstruct,len(self.grNames),len(self.objNames),len(self.tabNames),
                                epochs_inst=epochs_inst, n_layers_inst=n_layers_inst, size_inst=n_size_inst,
                                epochs_grasp=epochs_grasp, n_layers_grasp=n_layers_grasp, size_grasp=n_size_grasp,
                                epochs_place=epochs_place, n_layers_place=n_layers_place, size_place=n_size_place,
                                lr=lr, lr_drop=lr_drop, epoch_drop=epoch_drop, val_split=val_split, mode=self.dataMode,
                                listLog=self.listLog, reg0=reg0
                                )

            modelInstructHist, modelGraspHist, modelPlaceHist=self.rai_net.train(self.path_rai, model_dir)
            self.model_dir=self.rai_net.timestamp

            return modelInstructHist, modelGraspHist, modelPlaceHist

        elif self.NNmode in ["mixed10"]:
            modeMixed=self.dataMode
            input_size = (self.numGoalInstruct + len(self.objNames)+len(self.tabNames))*self.numGoal + len(self.logicalNames)*3

            self.prevInstr = np.zeros((5,input_size))
            self.input_size=input_size
            self.step=0
            

            self.rai_net=NNencoder.ClassifierMixed()

            self.rai_net.build_net(input_size,self.numActInstruct,len(self.logicalNames),
                                epochs_inst=epochs_inst, n_layers_inst=n_layers_inst, size_inst=n_size_inst,
                                epochs_grasp=epochs_grasp, n_layers_grasp=n_layers_grasp, size_grasp=n_size_grasp,
                                epochs_place=epochs_place, n_layers_place=n_layers_place, size_place=n_size_place,
                                lr=lr, lr_drop=lr_drop, epoch_drop=epoch_drop, clipnorm=clipnorm, val_split=val_split,
                                reg=reg, listLog=self.listLog, num_batch_it=num_batch_it, mode=modeMixed,
                                n_layers_inst2=n_layers_inst2, reg0=reg0
                                )
            modelInstructHist, modelGraspObjHist, modelGraspGrpHist, modelPlaceObjHist, modelPlaceGrpHist=self.rai_net.train(self.path_rai, model_dir, num_batch_it)
            self.model_dir=self.rai_net.timestamp.split("_")[0]

        elif self.NNmode=="FFnew":

            input_size = (self.numGoalInstruct + len(self.listLog[1])+len(self.listLog[2]))*self.numGoal + len(self.logicalNames)*3
            self.rai_net=NNencoder.ClassifierChainNew()

            self.rai_net.build_net(input_size,self.numActInstruct,len(self.logicalNames),
                                epochs_inst=epochs_inst, n_layers_inst=n_layers_inst, size_inst=n_size_inst,
                                epochs_grasp=epochs_grasp, n_layers_grasp=n_layers_grasp, size_grasp=n_size_grasp,
                                epochs_place=epochs_place, n_layers_place=n_layers_place, size_place=n_size_place,
                                lr=lr, lr_drop=lr_drop, epoch_drop=epoch_drop, clipnorm=clipnorm, val_split=val_split,
                                reg0=reg0, listLog=self.listLog, mode=self.dataMode
                                )
            modelInstructHist, modelGraspObjHist, modelGraspGrpHist, modelPlaceObjHist, modelPlaceGrpHist=self.rai_net.train(self.path_rai, model_dir)
            self.model_dir = self.rai_net.timestamp.split("_")[0]

        else:
            NotImplementedError


    def loadFit(self, model_dir=''):
        if self.NNmode=="minimal":
                self.rai_net = NNencoder.FeedForwardNN()
                self.rai_net.mode=self.dataMode
       
        elif self.NNmode in ["mixed10"]:
            self.rai_net=NNencoder.ClassifierMixed()
            self.rai_net.mode=self.dataMode
            self.input_size = (self.numGoalInstruct + len(self.objNames)+len(self.tabNames))*self.numGoal + len(self.logicalNames)*3
            #print(self.objNames, self.tabNames, self.logicalNames)
            self.prevInstr = np.zeros((5,self.input_size))
            self.step=0

        elif self.NNmode=="FFnew":
            self.rai_net=NNencoder.ClassifierChainNew()
            self.rai_net.mode=self.dataMode

        else:
            NotImplementedError
        self.rai_net.load_net(self.path_rai, model_dir)
        self.model_dir=model_dir


    def resetFit(self, cheatGoalState=False, goal=""):
        if self.NNmode in ["mixed10"]:
            self.prevInstr = np.zeros((5,self.input_size))
            self.step=0
        self.goalString=goal
        self.goalString_orig=goal
        self.goalState, _,_=self.preprocessGoalState(initState=False, cheatGoalState=cheatGoalState)

    #--------------Action Prediction: NN Softmax output to one hot, most probable action of all possible decisons-------------
    def padInput(self):
        #pad prevInput (to min len 2) to improve performance of cheat_goalstate mode
        if self.NNmode in ["mixed10"] and self.step==1:
            #inputState=self.prevInstr[self.step-1, : self.input_size]
            #self.prevInstr[self.step, : self.input_size] = inputState
            #self.step+=1

            self.prevInstr = np.zeros((5,self.input_size))
            self.step=0

    def processPrediction(self,inputState, oneHot=True):

        if self.setup=="minimal":

            goalState=inputState[:,:self.goallength]
            envState=inputState[:,self.goallength:]
            if self.NNmode=="minimal":
                instrPred = self.rai_net.modelInstruct.predict({"goal": goalState, "state": envState})

                if oneHot:
                    instrPred = softmaxToOnehot(instrPred)
                else:
                    instrPred=instrPred.reshape((1,-1))

                if instrPred[0,0]==1 or np.argmax(instrPred)==0: #grasp
                    graspPred=self.rai_net.modelGrasp.predict({"goal": goalState, "state": envState})

                    if oneHot:
                        graspPred=np.concatenate((softmaxToOnehot(graspPred[0]),softmaxToOnehot(graspPred[1])), axis=1)
                    else:
                        graspPred=np.concatenate((graspPred[0],graspPred[1]), axis=1)

                    return np.concatenate((instrPred,graspPred), axis=1)

                elif instrPred[0,1]==1 or np.argmax(instrPred)==1: #place
                    placePred=self.rai_net.modelPlace.predict({"goal": goalState, "state": envState})
                    if oneHot:
                        placePred=np.concatenate((softmaxToOnehot(placePred[0]),softmaxToOnehot(placePred[1]),softmaxToOnehot(placePred[2])), axis=1)
                    else:
                        placePred=np.concatenate((placePred[0],placePred[1],placePred[2]), axis=1)
                    
                    return np.concatenate((instrPred,placePred), axis=1)

                else:
                    NotImplementedError


            elif self.NNmode in ["mixed10"]:
                #print(inputState)
                if self.step <5:
                    self.prevInstr[self.step, : self.input_size] = inputState
                else:
                    self.prevInstr = np.concatenate((self.prevInstr[1:, : ], inputState), axis=0)
                self.step+=1

                instrPred = self.rai_net.modelInstruct.predict({"goal": goalState.reshape((1,1,-1)),
                                                                "state":self.prevInstr[0:4,self.goallength:].reshape((1,4, -1))})

                #print(self.prevInstr)
                #print(instrPred)

                if oneHot:
                    instrPred = softmaxToOnehot(instrPred)
                else:
                    instrPred=instrPred.reshape((1,-1))


                if instrPred[0,0]==1 or np.argmax(instrPred)==0: #grasp
                    graspPred=[None, None]
                    graspPred[1]=self.rai_net.modelGraspObj.predict({"goal": goalState, "state1": envState})

                    if oneHot:
                        graspPred[1]=softmaxToOnehot(graspPred[1])
                    
                    graspPred[0]=self.rai_net.modelGraspGrp.predict({"goal": goalState, "state2":np.concatenate((envState, graspPred[1]), axis=1)})

                    if oneHot:
                        graspPred[0]=softmaxToOnehot(graspPred[0])
                    
                    graspPred=np.concatenate((graspPred[0].reshape((1,-1)),graspPred[1].reshape((1,-1))), axis=1)

                    return np.concatenate((instrPred,graspPred), axis=1)
                    
                elif instrPred[0,1]==1 or np.argmax(instrPred)==1: #place
                    placePred=[None, None, None]
                    placePred[1]=self.rai_net.modelPlaceObj.predict({"goal": goalState, "state1": envState})

                    if oneHot:
                        placePred[1] = softmaxToOnehot(placePred[1])

                    [placePred[0], placePred[2]]=self.rai_net.modelPlaceGrpTab.predict({"goal": goalState, "state2":np.concatenate((envState, placePred[1]), axis=1)})

                    if oneHot:
                        placePred[0] = softmaxToOnehot(placePred[0])
                        placePred[2] = softmaxToOnehot(placePred[2])
                        
                    placePred=np.concatenate((placePred[0].reshape((1,-1)),placePred[1].reshape((1,-1)),placePred[2].reshape((1,-1))), axis=1)
                        
                    return np.concatenate((instrPred,placePred), axis=1)

                else:
                    NotImplementedError

            elif self.NNmode in ["FFnew"]:
                instrPred = self.rai_net.modelInstruct.predict({"goal": goalState, "state": envState})

                if oneHot:
                    instrPred = softmaxToOnehot(instrPred)
                else:
                    instrPred=instrPred.reshape((1,-1))

                if instrPred[0,0]==1 or np.argmax(instrPred)==0: #grasp
                    graspPred=[None, None]
                    graspPred[1]=self.rai_net.modelGraspObj.predict({"goal": goalState, "state1": envState})

                    if oneHot:
                        graspPred[1]=softmaxToOnehot(graspPred[1])
                    
                    graspPred[0]=self.rai_net.modelGraspGrp.predict({"goal": goalState, "state2":np.concatenate((envState, graspPred[1]), axis=1)})

                    if oneHot:
                        graspPred[0]=softmaxToOnehot(graspPred[0])
                    
                    graspPred=np.concatenate((graspPred[0].reshape((1,-1)),graspPred[1].reshape((1,-1))), axis=1)

                    return np.concatenate((instrPred,graspPred), axis=1)

                elif instrPred[0,1]==1 or np.argmax(instrPred)==1: #place
                    placePred=[None, None, None]
                    placePred[1]=self.rai_net.modelPlaceObj.predict({"goal": goalState, "state1": envState})

                    if oneHot:
                        placePred[1] = softmaxToOnehot(placePred[1])

                    [placePred[0], placePred[2]]=self.rai_net.modelPlaceGrpTab.predict({"goal": goalState, "state2":np.concatenate((envState, placePred[1]), axis=1)})

                    
                    if oneHot:
                        placePred[0] = softmaxToOnehot(placePred[0])
                        placePred[2] = softmaxToOnehot(placePred[2])
                        
                    placePred=np.concatenate((placePred[0].reshape((1,-1)),placePred[1].reshape((1,-1)),placePred[2].reshape((1,-1))), axis=1)
                        
                    return np.concatenate((instrPred,placePred), axis=1)

                else:
                    NotImplementedError

            else:
                NotImplementedError
           
        else:
            NotImplementedError

    

    def evalPredictions(self, inputState, infeasible=[""], prevSke="", depth=0):
        decisions=self.lgp.getDecisions()

        if self.NNmode=="mixed10":
            if "(place" in prevSke:
                mult=0.7
            else:
                mult=0.8
        else:
            mult=1

        #print(infeasible)

        if self.setup=="minimal":
            goalState=inputState[:,:self.goallength]
            envState=inputState[:,self.goallength:]

            if self.NNmode=="minimal":
                instrPred = self.rai_net.modelInstruct.predict({"goal": goalState, "state": envState})
                graspPred=self.rai_net.modelGrasp.predict({"goal": goalState, "state": envState})
                placePred=self.rai_net.modelPlace.predict({"goal": goalState, "state": envState})

            elif self.NNmode in ["mixed10"]:

                instrPred = self.rai_net.modelInstruct.predict({"goal": goalState.reshape((1,1,-1)),
                                                                    "state":self.prevInstr[0:4,self.goallength:].reshape((1,4, -1))})

                graspPred = [None, None]
                placePred = [None, None, None]

                graspPred[1]=self.rai_net.modelGraspObj.predict({"goal": goalState, "state1": envState})
                placePred[1]=self.rai_net.modelPlaceObj.predict({"goal": goalState, "state1": envState})

            
            elif self.NNmode in ["FFnew"]:
                graspPred = [None, None]
                placePred = [None, None, None]

                instrPred = self.rai_net.modelInstruct.predict({"goal": goalState, "state": envState})
                graspPred[1]=self.rai_net.modelGraspObj.predict({"goal": goalState, "state1": envState})
                placePred[1]=self.rai_net.modelPlaceObj.predict({"goal": goalState, "state1": envState})


            else:
                NotImplementedError
            
            instrPred[0,1]=instrPred[0,1]*mult
            instrPred[0,0]=1-instrPred[0,1]
            
            probBest=0
            actBest=""
            #print(decisions)
            for decision, i in zip(decisions, range(len(decisions))):
                validAct=True
                for log in splitCommandStep(decision, verbose=0):
                    if log in self.objNames_orig and not log in self.goalString:
                        validAct=False
                        continue
                
                if not validAct:
                    continue

                instructionEn, logEn = self.encodeAction(decision)

                if np.argmax(instructionEn, axis=1) == 0: #grasp gripper obj
                    graspPrev = logEn[:,len(self.grNames):len(self.grNames)+len(self.objNames)]
                    if self.NNmode in ["mixed10", "FFnew"]:
                        graspPred[0]=self.rai_net.modelGraspGrp.predict({"goal": goalState, "state2":np.concatenate((envState, graspPrev), axis=1)})

                    #graspPredProd=concatenate((0.3*graspPred[0],0.7*graspPred[1]), axis=1)
                    if self.NNmode in ["mixed10", "FFnew"]:
                        tmpProb=np.inner(np.concatenate((np.zeros((1,len(self.grNames))),graspPred[1]), axis=1), logEn)
                    else:
                        tmpProb=1
                    prob = np.inner(instrPred, instructionEn) + 2*np.inner(np.concatenate((0.1*tmpProb*graspPred[0],0.9*graspPred[1]), axis=1), logEn)
                    prob[0,0]=prob[0,0]/3

                elif np.argmax(instructionEn, axis=1) == 1 :#place gripper obj table
                    placePrev=[]
                    placePrev.append(logEn[:,len(self.grNames):len(self.grNames)+len(self.objNames)])
                    placePrev.append(logEn[:,:len(self.grNames)])
                    if self.checkPlaceSame(decision):
                        continue

                    if self.NNmode in ["FFnew","mixed10"]:
                        [placePred[0], placePred[2]]=self.rai_net.modelPlaceGrpTab.predict({"goal": goalState, "state2":np.concatenate((envState, placePrev[0]), axis=1)})

                    if self.NNmode in ["mixed10", "FFnew"]:
                        tmpProb=np.inner(np.concatenate((np.zeros((1,len(self.grNames))),placePred[1],np.zeros((1,len(self.tabNames)))), axis=1), logEn)
                    else:
                        tmpProb=1
                    #placePredProd=concatenate((0.1*placePred[0],0.6*placePred[1],0.3*placePred[2]), axis=1)
                    prob = np.inner(instrPred, instructionEn) + 2*np.inner(np.concatenate((0.1*tmpProb*placePred[0],0.45*placePred[1],tmpProb*0.45*placePred[2]), axis=1), logEn)
                    prob[0,0]=prob[0,0]/3

                if decision in infeasible:
                    prob=prob*(1.1-0.2*depth)

                print(decision, prob[0,0])
                
                if prob[0,0] > probBest:
                    actBest= decisions[i]
                    probBest=prob[0,0]

            return actBest, probBest
        else:
            NotImplementedError