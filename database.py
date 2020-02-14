import numpy as np
import rai_world
import minimal_experiment as expert

import sys
import os
import random

dir_file=os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir_file+'/../ry/')
from libry import *


import time
import datetime

skipTrain = []

numLogicalType=3
numGoal=2
numGoalInstruct=2
numActInstruct=2

numSets=72
numEnv=27
EnvAppend=[28,29,30,31,37,38,39,40,46,47,48,49,55,56,57,58,64,65,66,67,73,74,75,76,82,83,84,85,91,92,93,94]
EnvAppend2=[28,37,46,55,64,73,82,91,29,38,47,56,65,74,83,92,30,39,48,57,66,75,84,93,31,40,49,58,67,76,85,94]

def encodeAction(commandString, grNames, objNames, tabNames):
	instructionEn = np.zeros((1,numActInstruct))
	split_str = commandString.split(" ")

	if split_str[0] == '(grasp':
		instructionEn[(0,0)] = 1
		logEn = np.zeros((1, len(grNames)+len(objNames)+len(tabNames)))
		logEn[(0,grNames.index(split_str[1]))]=1

		split_str[2]=split_str[2][:-1]
		logEn[(0,len(grNames)+objNames.index(split_str[2]))]=1

	elif split_str[0] == '(place':
		instructionEn[(0,1)] = 1
		logEn = np.zeros((1, len(grNames)+len(objNames)+len(tabNames)))
		logEn[(0,grNames.index(split_str[1]))]=1

		split_str[2]=split_str[2]
		logEn[(0,len(grNames)+objNames.index(split_str[2]))]=1

		split_str[3]=split_str[3][:-1]
		logEn[(0,len(grNames)+len(objNames)+tabNames.index(split_str[3]))]=1

	else:
		logEn = np.zeros((1, 1))
		input("\nNot implemented")

	return instructionEn, logEn

def encodeAction3D(commandString, allNames): # !!!!! switches order: grasp object gripper, place object gripper table
    instructionEn = np.zeros((1,numActInstruct))
    prevLog = np.zeros((1, numLogicalType, numLogicalType, (numLogicalType-1)*len(allNames))) # prev for obj: 0 array
    split_str = commandString.split(" ")

    if split_str[0] == '(grasp':
        instructionEn[(0,0)] = 1
        logEn = np.zeros((1, numLogicalType,len(allNames)))

        #object
        split_str[2]=split_str[2][:-1]
        logEn[(0, 0, allNames.index(split_str[2]))]=1

        #gripper
        logEn[(0, 1, allNames.index(split_str[1]))]=1

        prevLog[0,1,1,:len(allNames)] = logEn[0,0,:] # prev for gripper: 0+obj decision

    elif split_str[0] == '(place':
        instructionEn[(0,1)] = 1
        logEn = np.zeros((1, numLogicalType,len(allNames)))

        #object
        split_str[2]=split_str[2]
        logEn[(0, 0, allNames.index(split_str[2]))]=1

        #gripper
        logEn[(0, 1, allNames.index(split_str[1]))]=1
        prevLog[0,1,1,:len(allNames)] = logEn[0,0,:] # prev for gripper: 0+obj decision

        #table
        split_str[3]=split_str[3][:-1]
        logEn[(0, 2, allNames.index(split_str[3]))]=1
        prevLog[0,2,1,:len(allNames)] = logEn[0,0,:]
        prevLog[0,2,2,len(allNames):] = logEn[0,1,:] # prev for tab: 0+obj+gripper decision


    else:
        logEn = np.zeros((1, 1))
        input("\nNot implemented")

    return instructionEn, logEn, prevLog

def dataSet(path_dB, rai, nenv, start,stop,append=False,mode=2):
    #X0 = rai.K.getFrameState()

    komo=None

    if not(os.path.exists(path_dB)):
        os.makedirs(path_dB)

    if rai.NNmode =="dataset":
        if not(os.path.exists(path_dB+'/env'+str(nenv).zfill(3))):
            os.makedirs(path_dB+'/env'+str(nenv).zfill(3))
    else:
        if not(os.path.exists(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode)):
            os.makedirs(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode)

    for nset in range(start,stop+1):
        if mode==2:
            solutions, goalString, numLoops = expert.getData(nset=nset, nenv=nenv)
        elif mode==1:
            solutions, goalString, numLoops = expert.getData1(nset=nset, nenv=nenv)

        if numLoops==0:
            print("No solution for env "+str(nenv)+" set "+str(nset))
            continue

        if rai.NNmode=="final" and os.path.isfile(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode+'/set'+str(nset).zfill(mode+1)+'Input.npy'):
            print("Already trained env "+str(nenv)+" set "+str(nset))
            continue

        if rai.NNmode=="minimal":
            input_size = (numGoalInstruct + len(rai.objNames)+len(rai.tabNames))*numGoal + len(rai.logicalNames)*(numLogicalType+3)
        elif rai.NNmode in ["full", "3d"]:
            input_size = (numGoalInstruct + len(rai.objNames)+len(rai.tabNames))*numGoal + len(rai.logicalNames)*(numLogicalType+3+4+4)
        elif rai.NNmode in ["dataset"]:
            input_size = (numGoalInstruct + len(rai.objNames)+len(rai.tabNames))*numGoal + len(rai.logicalNames)*2
        elif rai.NNmode in ["mixed0", "mixed3", "mixed2", "final"]:
            input_size = (numGoalInstruct + len(rai.objNames)+len(rai.tabNames))*numGoal + len(rai.logicalNames)*3
        else:
            NotImplementedError
            
        inputArray = np.zeros((numLoops,input_size), dtype=float)
        instrArray = np.zeros((numLoops,numActInstruct), dtype=int)

        if rai.NNmode in ["minimal", "full"]:
            logArray = np.zeros((numLoops,len(rai.grNames)+len(rai.objNames)+len(rai.tabNames)), dtype=int)
        elif rai.NNmode in ["3d", "dataset", "mixed2", "mixed0", "mixed3"]:
            logArray = np.zeros((numLoops,numLogicalType,len(rai.logicalNames)), dtype=int)
            prevInstrArray = np.zeros((numLoops,4,3*numActInstruct), dtype=int)
            prevInputArray = np.zeros((numLoops,4,input_size), dtype=float)
            prevLogArray = np.zeros((numLoops,numLogicalType, numLogicalType ,(numLogicalType-1)*len(rai.logicalNames)), dtype=int)
            if rai.NNmode in ["mixed3", "mixed2"]:
                feasibleArray = np.zeros((numLoops,2), dtype=int)  # act skeleton
        elif rai.NNmode == "final":
            logArray = np.zeros((numLoops,numLogicalType,len(rai.logicalNames)), dtype=int)
            prevInstrArray = np.zeros((numLoops,4,3*numActInstruct), dtype=int)
            prevLogArray = np.zeros((numLoops,numLogicalType, numLogicalType ,(numLogicalType-1)*len(rai.logicalNames)), dtype=int)

            prevInputArray = np.zeros((numLoops,4,input_size), dtype=float)
            feasibleArray = np.zeros((numLoops,2), dtype=int)  # act skeleton

            inputArray2 = np.zeros((numLoops,input_size), dtype=float)
            prevInputArray2 = np.zeros((numLoops,4,input_size), dtype=float)



        else:
            NotImplementedError


        i=0
        print("\n-------- Objective for set "+str(nset)+"/"+str(stop)+": "+goalString[0]+" --------")

        for sol in solutions:
            #rai.K.setFrameState(X0)
            rai.lgp.resetFrameState()
            commandList, commandStep=[], []
            commandList= rai_world.splitStringPath(sol, list_old=[],verbose=0)
            commandStep= rai_world.splitStringStep(sol, list_old=[],verbose=0)

            rai.lgp.walkToRoot()

            rai.lgp.walkToNode(commandList[-1],0)
            #rai.lgp.reset()
            rai.lgp.resetFrameState()
            komo = rai_world.runLGP(rai.lgp, BT.path, verbose=0, view=False)
            
            if not rai.NNmode in ["mixed3", "mixed2", "final"]:
                if rai.lgp.returnFeasible(BT.path):
                    rai.lgp.walkToRoot()
                    rai.lgp.resetFrameState()
                else:
                    rai.lgp.walkToRoot() 
                    print(sol+" infeasible")
                    continue

            print("env "+str(nenv)+", set "+str(nset)+": "+sol)
            i_start=i
            solfeas=0

            j=0
            previnst= np.zeros((1,3*numActInstruct), dtype=int)
            prevInput = np.zeros((1,4,input_size), dtype=float)

            if rai.NNmode=="final":
                prevInput2 = np.zeros((1,4,input_size), dtype=float)

            for command, clist in zip(commandStep,commandList):
                print('\t'+str(i+1)+"/"+str(numLoops)+" "+command)
                envState=rai.encodeState()

                for goal in goalString:                        
                    goalStep = rai_world.splitStringStep(goal, list_old=[],verbose=0)
                    goalState= rai.encodeGoal(goalStep)

                    if rai.NNmode in ["minimal", "full"]:
                            inputArray[i,:]=rai.encodeInput(envState, goalState=goalState)
                            try:
                                instrArray[i,:], logArray[i,:] = encodeAction(command,rai.grNames, rai.objNames, rai.tabNames)
                            except:
                                input("failed!: "+command+"\t"+clist)
                            i=i+1
                            #print(str(i)+"/"+str(numLoops)+"\n")
                    elif rai.NNmode in ["3d", "dataset", "mixed2", "mixed0", "mixed3", "final"]:

                            if rai.NNmode=="final":
                                [inputArray[i,:], inputArray2[i,:]]=rai.encodeInput(envState, goalState=goalState)
                            else:
                                inputArray[i,:]=rai.encodeInput(envState, goalState=goalState)

                            prevInput[0,j,:] = inputArray[i,:]
                            for k in range(0,j):
                                #print("j = ",j, "k = ", k)
                                prevInstrArray[i,k,:k*numActInstruct] = previnst[:,:k*numActInstruct]

                            prevInstrArray[i,j,:]=previnst
                            prevInputArray[i,:,:]=prevInput[0,:,:]

                            if rai.NNmode=="final":
                                prevInput2[0,j,:] = inputArray2[i,:]
                                prevInputArray2[i,:,:]=prevInput2[0,:,:]

                            try:
                                instrArray[i,:], logArray[i,:,:], prevLogArray[i,:,:,:] = encodeAction3D(command,rai.logicalNames)

                                if j<3 and i%mode==mode-1:
                                    previnst[0,j*numActInstruct:(j+1)*numActInstruct]=instrArray[i,:]
                                    j=j+1
                            except:
                                input("failed!: "+command+"\t"+clist)

                            #print(i, prevInputArray[i:i+1,:,:])
                            #print(prevInstrArray[i:i+1,:,:])
                            #input("test")
                            i=i+1
                            #print(str(i)+"/"+str(numLoops)+"\n")

                rai.lgp.walkToRoot()
                rai.lgp.walkToNode(clist,0)
                #rai.K.setFrameState(X0, verb=0)
                rai.lgp.resetFrameState()
                if rai.NNmode=="final":
                    komo = rai_world.runLGP(rai.lgp, BT.seq, verbose=0, view=False)
                    komo = rai_world.runLGP(rai.lgp, BT.seqPath, verbose=0, view=False)
                    if rai.lgp.returnFeasible(BT.seqPath):
                        feasibleArray[i-1,0]=1
                        if mode==2:
                            feasibleArray[i-2,0]=1

                        if clist==commandList[-1]:
                            solfeas=1

                else:
                    komo = rai_world.runLGP(rai.lgp, BT.path, verbose=0, view=False)
                    if rai.NNmode in ["mixed3", "mixed2"]:
                        if rai.lgp.returnFeasible(BT.path):
                            feasibleArray[i-1,0]=1
                            if mode==2:
                                feasibleArray[i-2,0]=1

                            if clist==commandList[-1]:
                                solfeas=1

                envState, config = rai_world.applyKomo(komo, rai.logicalNames, num=komo.getPathFrames(rai.logicalNames).shape[0]-1, verbose=0)
                rai.K.setFrameState(config, verb=0)
                #time.sleep(0.05)

            if rai.NNmode in ["mixed3", "mixed2", "final"]:
                feasibleArray[i_start:i,1]=solfeas
            
            #input(feasibleArray)
            #input(prevInputArray[0,:,:])
            #input(prevInputArray[2,:,:])
            #input(prevInputArray[4,:,:])
            #input(prevInputArray[6,:,:])

            #input(inputArray)
            #print(inputArray)
            #print(inputArray2)
            #print(prevInputArray)
            #print(prevInputArray2)
            #input(feasibleArray)

            #print(instrArray[i_old:i,:])
            #print(logArray[i_old:i,:,:])
            #print(prevLogArray[i_old:i,:,:,:])

            print("")

        if i==0:
            print("No feasible solution for env "+str(nenv)+" set "+str(nset))
            continue
        
        if rai.NNmode in ["mixed3", "mixed2", "final"]:
            print(str((np.where(~feasibleArray[:,1:2].any(axis=1))[0].shape[0])/mode)+" infeasible solutions")
        else:
            print(str((numLoops-i)/mode)+" infeasible solutions")
        #print(instrArray)
        #print(instrArray.shape)

        inputArray=np.delete(inputArray, list(range(i, numLoops)), axis=0)
        instrArray=np.delete(instrArray, list(range(i, numLoops)), axis=0)
        logArray=np.delete(logArray, list(range(i, numLoops)), axis=0)
        #print(instrArray.shape)
        #print(instrArray)
        #input("end")

        if rai.NNmode in ["3d", "dataset", "mixed2", "mixed0", "mixed3", "final"]:
            prevInstrArray=np.delete(prevInstrArray, list(range(i, numLoops)), axis=0)
            prevLogArray=np.delete(prevLogArray, list(range(i, numLoops)), axis=0)
            prevInputArray=np.delete(prevInputArray, list(range(i, numLoops)), axis=0)
            if rai.NNmode=="final":
                inputArray2=np.delete(inputArray, list(range(i, numLoops)), axis=0)
                prevInputArray2=np.delete(prevInputArray, list(range(i, numLoops)), axis=0)

        if append: #TODO
            old1=np.load(path_dB+'/input.npy')
            old2=np.load(path_dB+'/instruction.npy')
            old3=np.load(path_dB+'/logicals.npy')
        
            np.save(path_dB+'/input',np.concatenate((old1,inputArray), axis=0))
            np.save(path_dB+'/instruction', np.concatenate((old2,instrArray), axis=0))
            np.save(path_dB+'/logicals', np.concatenate((old3,logArray), axis=0))

        elif rai.NNmode in ["dataset"]:
            if mode==2:
                tmpZero=3
            elif mode==1:
                tmpZero=2
            np.save(path_dB+'/env'+str(nenv).zfill(3)+'/set'+str(nset).zfill(tmpZero)+'Input',inputArray)
            np.save(path_dB+'/env'+str(nenv).zfill(3)+'/set'+str(nset).zfill(tmpZero)+'Instruction', instrArray)
            np.save(path_dB+'/env'+str(nenv).zfill(3)+'/set'+str(nset).zfill(tmpZero)+'Logicals', logArray)

            np.save(path_dB+'/env'+str(nenv).zfill(3)+'/set'+str(nset).zfill(tmpZero)+'InstructionPrev', prevInstrArray)
            np.save(path_dB+'/env'+str(nenv).zfill(3)+'/set'+str(nset).zfill(tmpZero)+'InputPrev', prevInputArray)
            np.save(path_dB+'/env'+str(nenv).zfill(3)+'/set'+str(nset).zfill(tmpZero)+'LogicalsPrev', prevLogArray)

        else:
            if mode==2:
                tmpZero=3
            elif mode==1:
                tmpZero=2
            np.save(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode+'/set'+str(nset).zfill(tmpZero)+'Input',inputArray)
            np.save(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode+'/set'+str(nset).zfill(tmpZero)+'Instruction', instrArray)
            np.save(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode+'/set'+str(nset).zfill(tmpZero)+'Logicals', logArray)

            if rai.NNmode in ["3d", "mixed2", "mixed0", "mixed3", "final"]:
                #np.save(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode+'/set'+str(nset).zfill(tmpZero)+'InstructionPrev', prevInstrArray)
                np.save(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode+'/set'+str(nset).zfill(tmpZero)+'InputPrev', prevInputArray)
                #np.save(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode+'/set'+str(nset).zfill(tmpZero)+'LogicalsPrev', prevLogArray)
                if rai.NNmode in ["mixed3", "mixed2", "final"]:
                    np.save(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode+'/set'+str(nset).zfill(tmpZero)+'Feasible', feasibleArray)
                    if rai.NNmode=="final":
                        np.save(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode+'/set'+str(nset).zfill(tmpZero)+'Input2',inputArray2)
                        np.save(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode+'/set'+str(nset).zfill(tmpZero)+'InputPrev2', prevInputArray2)

def offsetDataSet(Input, Logicals, prevInput, Input2, prevInput2, mode=2, feasible=False):
    finalInput=Input
    finalLogicals=Logicals
    finalprevInput=prevInput
    finalInput2=Input2
    finalprevInput2=prevInput2

    grNames=[0,1]
    objNames=[2,3,4]
    tabNames=[2,3,4,5,6]
    logicalNames=[0,1,2,3,4,5,6]

    input_size = (numGoalInstruct + len(objNames)+len(tabNames))*numGoal + len(logicalNames)*3
    
    if mode==2:
        switches=[[0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]

    else:
        switches=[[2,0,1], [1,2,0]]

    for switch in switches:
        Inputidx=(list(range(numGoalInstruct))+[x+numGoalInstruct for x in switch]+[x+numGoalInstruct+len(objNames) for x in switch]+list(range( numGoalInstruct + 2*len(objNames), 2*numGoalInstruct + len(objNames)+len(tabNames)))+
                    [x+2*numGoalInstruct +len(objNames)+len(tabNames) for x in switch] + [x+2*numGoalInstruct +2*len(objNames)+len(tabNames) for x in switch] + 
                    list(range(2*numGoalInstruct + 3*len(objNames)+len(tabNames), 2*(numGoalInstruct + len(objNames)+len(tabNames))+len(grNames)*3)) +
                    [x+2*(numGoalInstruct + len(objNames) + len(tabNames))+3*(len(grNames) + switch[0]) for x in [0,1,2]]+
                    [x+2*(numGoalInstruct + len(objNames) + len(tabNames))+3*(len(grNames) + switch[1]) for x in [0,1,2]]+
                    [x+2*(numGoalInstruct + len(objNames) + len(tabNames))+3*(len(grNames) + switch[2]) for x in [0,1,2]]+
                    #[x+2*(numGoalInstruct + len(objNames) + len(tabNames))+3*(len(grNames) + len(objNames) + switch[0]) for x in [0,1,2]]+
                    #[x+2*(numGoalInstruct + len(objNames) + len(tabNames))+3*(len(grNames) + len(objNames) + switch[1]) for x in [0,1,2]]+
                    #[x+2*(numGoalInstruct + len(objNames) + len(tabNames))+3*(len(grNames) + len(objNames) + switch[2]) for x in [0,1,2]]+
                    list(range(2*(numGoalInstruct + len(objNames)+len(tabNames))+3*(len(grNames)+len(objNames)), input_size)) 
                    )
        finalInput = np.concatenate((finalInput, Input[:, Inputidx]), axis=0)
        finalInput2= np.concatenate((finalInput2, Input2[:, Inputidx]), axis=0)
        finalprevInput =np.concatenate((finalprevInput,prevInput[:, :, Inputidx]), axis=0)
        finalprevInput2=np.concatenate((finalprevInput2,prevInput2[:, :, Inputidx]), axis=0)

        Logidx=list(range(len(grNames)))+[x + len(grNames) for x in switch]+list(range(len(grNames)+len(objNames), len(logicalNames)))
        finalLogicals = np.concatenate((finalLogicals, Logicals[:,:,Logidx]), axis=0)

        #print(Inputidx)
        #print(finalInput)
        #print(Logidx)
        #print(finalLogicals)
        #input(switch)
    
    if Input.shape[0]<49 and mode==2 and not feasible:
        finalInput = np.concatenate((finalInput, finalInput), axis=0)
        finalInput2= np.concatenate((finalInput2, finalInput2), axis=0)
        finalprevInput =np.concatenate((finalprevInput,finalprevInput), axis=0)
        finalprevInput2=np.concatenate((finalprevInput2,finalprevInput2), axis=0)

        finalLogicals = np.concatenate((finalLogicals, finalLogicals), axis=0)


    return finalInput, finalLogicals, finalprevInput, finalInput2, finalprevInput2

def concatData(path_dB,start,stop, skip1=False, rand2=0, NNmode="minimal", mixData=False, feasible=False):

    if rand2>0 and rand2<numSets:
        arrSets=np.arange(1,numSets+1)
        np.random.shuffle(arrSets)
        arrSets=arrSets[0:rand2]
    else:
        arrSets=np.arange(start,stop+1)
    
    if feasible:
        arrSets=[6,7,12,20,21,27,32,45,52,61,62,67,68,71,4,51,54,1]
        arrSets0=[x for x in list(range(1,numSets+1)) if x not in arrSets]
        random.shuffle(arrSets0)
        arrSets=arrSets+arrSets0[:15]
        envSets=[17,18,19,25,28,40,46,56,64,76,84,93]
    elif mixData:
        arrSets=[6,7,12,20,21,27,32,45,52,61,62,67,68,71,4,51,54,1]
                #13,55,28]
        #arrSets=[6,7,12,20,21,27,32,45,52,61,62,67,68,71]#,7,21,52,6]
        if rand2>0 and not rand2==len(arrSets):
            if rand2>len(arrSets):
                arrSets0=[x for x in list(range(1,numSets+1)) if x not in arrSets]
                random.shuffle(arrSets0)
                arrSets=arrSets+arrSets0[:rand2-len(arrSets)]
            else:
                random.shuffle(arrSets)
                arrSets=arrSets[:rand2]
            input(arrSets)
        
        #envSets=[7, 30, 31, 37, 39, 46, 48, 55, 58, 66, 70, 76, 67, 83, 85, 92, 94]
        #envSets=[30, 35, 38, 41, 48, 49, 57, 58, 65, 70, 76, 79, 82, 83, 92, 94, 31, 91, 29, 93]#, 22, 25]
        envSets=[5,7,9,11,15,17,21,23,26,30,31,36,37,41,45,47,52,53,55,61,62,67,69,70,74,76,79,85,86,87,92,98,99, 
                6,14,27,28,44,49,60,71,77,84,95]
    else:
        #arrSets=[46,59,36,11,13,40,57,19,20,35,65,28,23,42,25,66,30,53,18,7,39,44,24,54,31,16,56,71,21,45,34,49,4,17,22,26,38,29,10,68,62,27,41,9,70]
        arrSets=[46,59,36,11,13,40,57,19,20,35,65,28,23,42,25,66,30,53,18,7,39,44,24,54,31,16,56,71,21,45,34,49,4,37,22,26,38,29,10,68,62,27,15,9,60]
        #envSets=list(range(1,numEnv+1)) + EnvAppend
        envSets=list(range(1,18+1))+[30,31,36,37,41,45,47,52,53,55,61,62,67,69,70,74,76,79,85,86,87,92,98,99, 
                27,28,44,49,60,71,77,84,95,
                34,96]

    now=datetime.datetime.now()
    timestamp=str(now.year)+str(now.month).zfill(2)+str(now.day).zfill(2)+"-"+str(now.hour).zfill(2)+str(now.minute).zfill(2)+str(now.second).zfill(2)

    if NNmode=="final":
        path_dB=path_dB+"_new"

    if NNmode=="dataset":
        appendName=""
    else:
        appendName="_"+NNmode


    listEnv, listNoEnv=[],[]

    for i in arrSets:
        if not os.path.isfile(path_dB+'/env'+str(envSets[0]).zfill(3)+appendName+'/set'+str(i).zfill(3)+'Input.npy'):
            continue

        start=i
        old1=np.load(path_dB+'/env'+str(envSets[0]).zfill(3)+appendName+'/set'+str(start).zfill(3)+'Input.npy')
        old2=np.load(path_dB+'/env'+str(envSets[0]).zfill(3)+appendName+'/set'+str(start).zfill(3)+'Instruction.npy')
        old3=np.load(path_dB+'/env'+str(envSets[0]).zfill(3)+appendName+'/set'+str(start).zfill(3)+'Logicals.npy')

        if NNmode in ["3d", "dataset", "mixed2", "mixed0", "mixed3", "final"]:
            #old4=np.load(path_dB+'/env'+str(envSets[0]).zfill(3)+appendName+'/set'+str(start).zfill(3)+'InstructionPrev.npy')
            #old5=np.load(path_dB+'/env'+str(envSets[0]).zfill(3)+appendName+'/set'+str(start).zfill(3)+'LogicalsPrev.npy')
            old6=np.load(path_dB+'/env'+str(envSets[0]).zfill(3)+appendName+'/set'+str(start).zfill(3)+'InputPrev.npy')
            if NNmode in ["mixed3", "mixed2","final"]:
                old7=np.load(path_dB+'/env'+str(envSets[0]).zfill(3)+appendName+'/set'+str(start).zfill(3)+'Feasible.npy')
            if NNmode=="final":
                old8=np.load(path_dB+'/env'+str(envSets[0]).zfill(3)+appendName+'/set'+str(start).zfill(3)+'Input2.npy')
                old9=np.load(path_dB+'/env'+str(envSets[0]).zfill(3)+appendName+'/set'+str(start).zfill(3)+'InputPrev2.npy')

                if mixData:
                    old1, old3, old6, old8, old9 = offsetDataSet(old1, old3, old6, old8, old9, feasible=feasible)
                    old2= np.tile(old2,(6,1))
                    #old4= np.tile(old4,(6,1,1))
                    old7= np.tile(old7,(6,1))
                    if old2.shape[0]<50*6 and not feasible:
                        old2=np.concatenate((old2,old2), axis=0)
                        #old4=np.concatenate((old4,old4), axis=0)
                        old7=np.concatenate((old7,old7), axis=0)
                elif old2.shape[0]<49:
                    old1=np.concatenate((old1,old1), axis=0)
                    old2=np.concatenate((old2,old2), axis=0)
                    old3=np.concatenate((old3,old3), axis=0)
                    old6=np.concatenate((old6,old6), axis=0)
                    old7=np.concatenate((old7,old7), axis=0)
                    old8=np.concatenate((old8,old8), axis=0)
                    old9=np.concatenate((old9,old9), axis=0)


        
        break


    for nenv in envSets:

        if (nenv in skipTrain) or not(os.path.exists(path_dB+'/env'+str(nenv).zfill(3)+appendName)):
            print("skip env "+str(nenv))
            continue
        else:
            print("process env "+str(nenv))
        
        print(expert.getEnvInfo(nenv,"r"),expert.getEnvInfo(nenv,"g"),expert.getEnvInfo(nenv,"b"))

        listSet, listNoSet=[],[]
        listSet.append(str(nenv).zfill(3))
        listNoSet.append(str(nenv).zfill(3))

        for nset in arrSets:
            if nenv is envSets[0] and nset is start and not nset in listSet:
                listSet.append(str(nset).zfill(3))
                continue
            
            if mixData and random.random()>0.5:
                if nset==32 and expert.getEnvInfo(nenv,"g")==1:
                    nset=33
                elif nset==71 and expert.getEnvInfo(nenv,"b")==2:
                    nset==70
                elif nset==20 and expert.getEnvInfo(nenv,"r")==1:
                    nset=30
                elif nset==7 and expert.getEnvInfo(nenv,"b")==1:
                    nset=8
                elif nset==21 and expert.getEnvInfo(nenv,"r")==2:
                    nset=11
                elif nset==54 and expert.getEnvInfo(nenv,"g")==1:
                    nset=59
                elif nset==51 and expert.getEnvInfo(nenv,"b")==2:
                    nset==50
                elif nset==62 and expert.getEnvInfo(nenv,"g")==2:
                    nset=57
                elif nset==12 and (expert.getEnvInfo(nenv,"g")==1 and expert.getEnvInfo(nenv,"r")==1):
                    if expert.getEnvInfo(nenv,"g")==2:
                        nset=17
                    else:
                        nset=55
                elif nset==61 and (expert.getEnvInfo(nenv,"g")==2 and expert.getEnvInfo(nenv,"b")==2):
                    if expert.getEnvInfo(nenv,"g")==1:
                        nset=28
                    else:
                        nset=23
                elif nset==27 and (expert.getEnvInfo(nenv,"b")==1 and expert.getEnvInfo(nenv,"r")==2):
                    if expert.getEnvInfo(nenv,"g")==2 and expert.getEnvInfo(nenv,"r")==1:
                        nset=22
                    elif expert.getEnvInfo(nenv,"g")==2 and expert.getEnvInfo(nenv,"b")==1:
                        nset=56
                    elif expert.getEnvInfo(nenv,"r")==2 and expert.getEnvInfo(nenv,"b")==1:
                        nset=18
                    elif expert.getEnvInfo(nenv,"r")==2 and expert.getEnvInfo(nenv,"g")==1:
                        nset=13
                    elif expert.getEnvInfo(nenv,"b")==2 and expert.getEnvInfo(nenv,"g")==1:
                        nset=60

            if not os.path.isfile(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(3)+'Input.npy'):
                print("skip    env "+str(nenv)+" set "+str(nset).zfill(3))
                listNoSet.append(str(nset).zfill(3))
                continue

            #print("process env "+str(nenv)+" set "+str(nset).zfill(3))

            new1=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(3)+'Input.npy')
            new2=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(3)+'Instruction.npy')
            new3=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(3)+'Logicals.npy')

            if NNmode in ["3d", "dataset", "mixed2", "mixed0", "mixed3", "final"]:
                #new4=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(3)+'InstructionPrev.npy')
                #new5=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(3)+'LogicalsPrev.npy')
                new6=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(3)+'InputPrev.npy')
                if NNmode in ["mixed3", "mixed2", "final"]:
                    new7=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(3)+'Feasible.npy')
                    if NNmode=="final":
                        new8=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(3)+'Input2.npy')
                        new9=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(3)+'InputPrev2.npy')

                        if mixData:
                            new1, new3, new6, new8, new9 = offsetDataSet(new1, new3, new6, new8, new9, feasible=feasible)
                            new2= np.tile(new2,(6,1))
                            #new4= np.tile(new4,(6,1,1))
                            new7= np.tile(new7,(6,1))
                            #print(new2.shape)
                            if new2.shape[0]<48*6+1 and not feasible:
                                new2=np.concatenate((new2,new2), axis=0)
                                #new4=np.concatenate((new4,new4), axis=0)
                                new7=np.concatenate((new7,new7), axis=0)
                        elif new2.shape[0]<49:
                            new1=np.concatenate((new1,new1), axis=0)
                            new2=np.concatenate((new2,new2), axis=0)
                            new3=np.concatenate((new3,new3), axis=0)
                            new6=np.concatenate((new6,new6), axis=0)
                            new7=np.concatenate((new7,new7), axis=0)
                            new8=np.concatenate((new8,new8), axis=0)
                            new9=np.concatenate((new9,new9), axis=0)

                        if NNmode in ["final"]:
                            try:
                                assert new1.shape[0] == new2.shape[0], "Inconsistent dimensions InstructionPrev"+str(new1.shape[0])+" != "+str(new2.shape[0])
                                assert new1.shape[0] == new3.shape[0], "Inconsistent dimensions LogicalsPrev"+str(new1.shape[0])+" != "+str(new3.shape[0])
                                assert new1.shape[0] == new6.shape[0], "Inconsistent dimensions InputPrev"+str(new1.shape[0])+" != "+str(new6.shape[0])
                                assert new1.shape[0] == new7.shape[0], "Inconsistent dimensions Feasible"+str(new1.shape[0])+" != "+str(new7.shape[0])
                    
                                assert new1.shape[0] == new8.shape[0], "Inconsistent dimensions Input2"+str(new1.shape[0])+" != "+str(new8.shape[0])
                                assert new1.shape[0] == new9.shape[0], "Inconsistent dimensions InputPrev2"+str(new1.shape[0])+" != "+str(new9.shape[0])
                            except:
                                print("skip    env "+str(nenv)+" set "+str(nset).zfill(3)+" as inconsistent")
                                listNoSet.append(str(nset).zfill(3))
                                continue

                        old8=np.concatenate((old8,new8), axis=0)
                        old9=np.concatenate((old9,new9), axis=0)

                    old7=np.concatenate((old7,new7), axis=0)

                #old4=np.concatenate((old4,new4), axis=0)
                #old5=np.concatenate((old5,new5), axis=0)
                old6=np.concatenate((old6,new6), axis=0)

            old1=np.concatenate((old1,new1), axis=0)
            old2=np.concatenate((old2,new2), axis=0)
            old3=np.concatenate((old3,new3), axis=0)

            if not str(nset).zfill(3) in listSet:
                listSet.append(str(nset).zfill(3))
        
        if not skip1:
            listSet.append("|")
            if mixData:
                set1_tmp=[[1,2,3,4,15],[6,7,8,10,5],[11,12,13,14,9]]
                set1 = set1_tmp[random.randint(0,2)]

            else:
                set1=range(1,16)
            for nset in set1:

                if not os.path.isfile(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(2)+'Input.npy'):
                    print("skip    env "+str(nenv)+" set "+str(nset).zfill(2))
                    listNoSet.append(str(nset).zfill(2))
                    continue

                #print("process env "+str(nenv)+" set "+str(nset).zfill(2))
                new1=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(2)+'Input.npy')
                new2=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(2)+'Instruction.npy')
                new3=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(2)+'Logicals.npy')

                if NNmode in ["3d", "dataset", "mixed2", "mixed0", "mixed3","final"]:
                    #new4=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(2)+'InstructionPrev.npy')
                    #new5=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(2)+'LogicalsPrev.npy')
                    new6=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(2)+'InputPrev.npy')
                    if NNmode in ["mixed3", "mixed2","final"]:
                        new7=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(2)+'Feasible.npy')
                        
                        if NNmode=="final":
                            new8=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(2)+'Input2.npy')
                            new9=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(2)+'InputPrev2.npy')

                            if mixData:
                                new1, new3, new6, new8, new9 = offsetDataSet(new1, new3, new6, new8, new9, mode=1)
                                new2= np.tile(new2,(3,1))
                                #new4= np.tile(new4,(3,1,1))
                                new7= np.tile(new7,(3,1))

                            old8=np.concatenate((old8,new8), axis=0)
                            old9=np.concatenate((old9,new9), axis=0)
                            
                        old7=np.concatenate((old7,new7), axis=0)

                    #old4=np.concatenate((old4,new4), axis=0)
                    #old5=np.concatenate((old5,new5), axis=0)
                    old6=np.concatenate((old6,new6), axis=0)

                old1=np.concatenate((old1,new1), axis=0)
                old2=np.concatenate((old2,new2), axis=0)
                old3=np.concatenate((old3,new3), axis=0)
                listSet.append(str(nset).zfill(2))
            
        listEnv.append(listSet)
        listNoEnv.append(listNoSet)
        print("")

    #correct datasets
    #idx = np.all(old6[:,0,:]==0, axis=1)
    #old6=old6[~idx,:,:]


    if NNmode in ["mixed3", "mixed2", "final"]:
        infeasible=np.where(~old7[:,1:2].any(axis=1))[0]
        feasible=np.where(old7[:,1:2].any(axis=1))[0]
        new1=old1 #input
        new2=old2 #instr
        new3=old3 #log
        #new4=old4 #instrprev   
        #new5=old5 #logprev
        new6=old6 #inputprev
        new7=old7 #feas
        old1=old1[feasible,:]
        old2=old2[feasible,:]
        old3=old3[feasible,:,:]
        #old4=old4[feasible,:,:]
        #old5=old5[feasible,:,:,:]
        old6=old6[feasible,:,:]
        old7=old7[feasible,:]
        new1=new1[infeasible,:] #input
        new2=new2[infeasible,:] #instr
        new3=new3[infeasible,:,:] #log
        #new4=new4[infeasible,:,:] #instrprev
        #new5=new5[infeasible,:,:,:] #logprev
        new6=new6[infeasible,:,:] #inputprev
        new7=new7[infeasible,:] #infeas
        if NNmode=="final":
            new8=old8
            new9=old9
            old8=old8[feasible,:]
            new8=new8[infeasible,:]
            old9=old9[feasible,:,:]
            new9=new9[infeasible,:,:]


    assert old1.shape[0] == old2.shape[0], "Inconsistent dimensions Instruction: "+str(old1.shape)[0]+" != "+str(old2.shape[0])
    assert old1.shape[0] == old3.shape[0], "Inconsistent dimensions Logicals"+str(old1.shape)[0]+" != "+str(old3.shape[0])

    if NNmode in ["3d", "dataset", "mixed2", "mixed0", "mixed3", "final"]:
        #assert old1.shape[0] == old4.shape[0], "Inconsistent dimensions InstructionPrev"+str(old1.shape)[0]+" != "+str(old4.shape[0])
        #assert old1.shape[0] == old5.shape[0], "Inconsistent dimensions LogicalsPrev"+str(old1.shape)[0]+" != "+str(old5.shape[0])
        assert old1.shape[0] == old6.shape[0], "Inconsistent dimensions InputPrev"+str(old1.shape)[0]+" != "+str(old6.shape[0])
        if NNmode in ["mixed3", "mixed2","final"]:
            assert old1.shape[0] == old7.shape[0], "Inconsistent dimensions Feasible"+str(old1.shape)[0]+" != "+str(old7.shape[0])
            assert new1.shape[0] == new7.shape[0], "Inconsistent dimensions Infeasible"+str(new1.shape)[0]+" != "+str(old7.shape[0])
            assert new1.shape[0] == new2.shape[0], "Inconsistent dimensions InfeasibleInstr"+str(new1.shape)[0]+" != "+str(new2.shape[0])
            assert new1.shape[0] == new3.shape[0], "Inconsistent dimensions InfeasibleLog"+str(new1.shape)[0]+" != "+str(new3.shape[0])
            #assert new1.shape[0] == new4.shape[0], "Inconsistent dimensions InfeasibleInstrPrev"+str(old1.shape)[0]+" != "+str(new4.shape[0])
            #assert new1.shape[0] == new5.shape[0], "Inconsistent dimensions InfeasibleLogPrev"+str(old1.shape)[0]+" != "+str(new5.shape[0])
            assert new1.shape[0] == new6.shape[0], "Inconsistent dimensions InfeasibleInputPrev"+str(old1.shape)[0]+" != "+str(new6.shape[0])
            if NNmode=="final":
                assert old1.shape[0] == old8.shape[0], "Inconsistent dimensions Input2"+str(old1.shape)[0]+" != "+str(old8.shape[0])
                assert new1.shape[0] == new8.shape[0], "Inconsistent dimensions InfeasibleInput2"+str(new1.shape)[0]+" != "+str(old8.shape[0])
                assert old1.shape[0] == old9.shape[0], "Inconsistent dimensions InputPrev2"+str(old1.shape)[0]+" != "+str(old9.shape[0])
                assert new1.shape[0] == new9.shape[0], "Inconsistent dimensions InfeasibleInputPrev2"+str(new1.shape)[0]+" != "+str(old9.shape[0])

    if not(os.path.exists(path_dB+'/'+timestamp+appendName)):
        os.makedirs(path_dB+'/'+timestamp+appendName)
    np.save(path_dB+'/'+timestamp+appendName+'/Input',old1)
    np.save(path_dB+'/'+timestamp+appendName+'/Instruction', old2)
    np.save(path_dB+'/'+timestamp+appendName+'/Logicals', old3)

    if NNmode in ["3d", "dataset", "mixed2", "mixed0", "mixed3","final"]:
        if NNmode in ["mixed3", "mixed2","final"]:
            np.save(path_dB+'/'+timestamp+appendName+'/Feasible', old7)
            np.save(path_dB+'/'+timestamp+appendName+'/InputInfeasible', new1)
            np.save(path_dB+'/'+timestamp+appendName+'/InFeasible', new7)
            np.save(path_dB+'/'+timestamp+appendName+'/InFeasibleInstr', new2)
            np.save(path_dB+'/'+timestamp+appendName+'/InFeasibleLog', new3)
            #np.save(path_dB+'/'+timestamp+appendName+'/InFeasibleInstrPrev', new4)
            #np.save(path_dB+'/'+timestamp+appendName+'/InFeasibleLogPrev', new5)
            np.save(path_dB+'/'+timestamp+appendName+'/InFeasibleInputPrev', new6)
            if NNmode=="final":
                np.save(path_dB+'/'+timestamp+appendName+'/Input_feat',old8)
                np.save(path_dB+'/'+timestamp+appendName+'/InputInfeasible_feat', new8)
                np.save(path_dB+'/'+timestamp+appendName+'/InFeasibleInputPrev_feat', new9)

        #np.save(path_dB+'/'+timestamp+appendName+'/InstructionPrev', old4)
        if old1.shape[0]>160000 and False:
            idxSplit=int(old1.shape[0]/3)
            #np.save(path_dB+'/'+timestamp+appendName+'/LogicalsPrev1', old5[:idxSplit,:,:,:])
            #np.save(path_dB+'/'+timestamp+appendName+'/LogicalsPrev2', old5[idxSplit:2*idxSplit,:,:,:])
            #np.save(path_dB+'/'+timestamp+appendName+'/LogicalsPrev3', old5[2*idxSplit:,:,:,:])
            np.save(path_dB+'/'+timestamp+appendName+'/InputPrev1', old6[:idxSplit,:,:])
            np.save(path_dB+'/'+timestamp+appendName+'/InputPrev2', old6[idxSplit:2*idxSplit,:,:])
            np.save(path_dB+'/'+timestamp+appendName+'/InputPrev3', old6[2*idxSplit:,:,:])
            if NNmode=="final":
                np.save(path_dB+'/'+timestamp+appendName+'/InputPrev1_feat', old9[:idxSplit,:,:])
                np.save(path_dB+'/'+timestamp+appendName+'/InputPrev2_feat', old9[idxSplit:2*idxSplit,:,:])
                np.save(path_dB+'/'+timestamp+appendName+'/InputPrev3_feat', old9[2*idxSplit:,:,:])

        elif old1.shape[0]>80000 and False:
            idxSplit=int(old1.shape[0]/2)
            #print(old5.shape)
            #print(old6.shape)
            #np.save(path_dB+'/'+timestamp+appendName+'/LogicalsPrev1', old5[:idxSplit,:,:,:])
            #np.save(path_dB+'/'+timestamp+appendName+'/LogicalsPrev2', old5[idxSplit:,:,:,:])
            np.save(path_dB+'/'+timestamp+appendName+'/InputPrev1', old6[:idxSplit,:,:])
            np.save(path_dB+'/'+timestamp+appendName+'/InputPrev2', old6[idxSplit:,:,:])
            if NNmode=="final":
                np.save(path_dB+'/'+timestamp+appendName+'/InputPrev1_feat', old9[:idxSplit,:,:])
                np.save(path_dB+'/'+timestamp+appendName+'/InputPrev2_feat', old9[idxSplit:,:,:])
        else:
            #np.save(path_dB+'/'+timestamp+appendName+'/LogicalsPrev', old5)
            np.save(path_dB+'/'+timestamp+appendName+'/InputPrev', old6)
            if NNmode=="final":
                np.save(path_dB+'/'+timestamp+appendName+'/InputPrev_feat', old9)


    with open(path_dB+'/'+timestamp+appendName+'/Sets.txt', 'w') as f:
        for item in listEnv:
            for i, it in zip(item,range(len(item))):
                if it==0:
                    f.write("env "+str(i).rjust(3)+":\t".expandtabs(4))
                else:
                    f.write(str(i).rjust(3)+"\t".expandtabs(2))
            f.write("\n")

    with open(path_dB+'/'+timestamp+appendName+'/NoSets.txt', 'w') as f:
        for item in listNoEnv:
            for i, it in zip(item,range(len(item))):
                if it==0:
                    f.write("env "+str(i).rjust(3)+":\t".expandtabs(4))
                else:
                    f.write(str(i).rjust(3)+"\t".expandtabs(2))
            f.write("\n")

    if NNmode in ["mixed3", "mixed2", "final"]:
        tmpstr=" feasible and "+str(new1.shape[0])+" infeasible"
    else:
        tmpstr=""
    print("---Finished dataset with "+str(old1.shape[0])+tmpstr+" samples---")

    with open(path_dB+'/'+timestamp+appendName+'/Summary.txt', 'w') as f:
        f.write("Dataset with "+str(old1.shape[0])+tmpstr+" samples\n\n")
        f.write("Environments:\n\t")
        for i in list(range(1,numEnv+1)) + EnvAppend:
            f.write(str(i)+"\t")
        f.write("\n\nSets:\n\t")
        for i in arrSets:
            f.write(str(i).zfill(3)+"\t")
        if not skip1:
            f.write("\n\t")
            for i in range(1,16):
                f.write(str(i).zfill(2)+"\t")





def main():

    #dir_file=os.path.abspath(os.path.dirname(__file__))


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rai_dir', type=str, default=dir_file)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--env', type=int, default=0)
    parser.add_argument('--start2', type=int, default=1)
    parser.add_argument('--stop2', type=int, default=numSets)
    parser.add_argument('--start1', type=int, default=1)
    parser.add_argument('--stop1', type=int, default=15)
    parser.add_argument('--setup', type=str, default="minimal")
    parser.add_argument('--NNmode', type=str, default="minimal")
    parser.add_argument('--datasetMode', type=int, default=1)
    parser.add_argument('--viewConfig', dest='viewConfig', action='store_true')
    parser.set_defaults(viewConfig=False)

    parser.add_argument('--feasible', dest='feasible', action='store_true')
    parser.set_defaults(feasible=False)

    parser.add_argument('--mixData', dest='mixData', action='store_true')
    parser.set_defaults(mixData=False)

    parser.add_argument('--rand2', type=int, default=0)

    parser.add_argument('--skip1', dest='skip1', action='store_true')
    parser.set_defaults(skip1=False)
    parser.add_argument('--skip2', dest='skip2', action='store_true')
    parser.set_defaults(skip2=False)
    
    args = parser.parse_args()
    path_rai = args.rai_dir
    verbose=args.verbose
    start2=args.start2
    stop2=args.stop2
    start1=args.start1
    stop1=args.stop1
    nenv=args.env
    setup=args.setup
    NNmode=args.NNmode
    dataMode=args.datasetMode
    feasible=args.feasible

    skip1=args.skip1
    skip2=args.skip2
    rand2=args.rand2
    viewConfig=args.viewConfig
    mixData=args.mixData
    
    """with open(path_rai+'/final2.txt', 'w') as f:
        for i in [28, 37, 46, 55, 64, 73, 82, 91, 29, 38, 47, 56, 65, 74, 83, 92, 30, 39, 48, 57, 66, 75, 84, 93, 31, 40, 49, 58, 67, 76, 85, 94, 32, 41, 50, 59, 68, 77, 86, 95, 33, 42, 51, 60, 69, 78, 87, 96, 34, 43, 52, 61, 70, 79, 88, 97, 35, 44, 53, 62, 71, 80, 89, 98, 36, 45, 54, 63, 72, 81, 90, 99]:
            f.write("python database.py --env="+str(i)+' --stop2=12 --NNmode="final"; python database.py --env='+str(i)+' --start2=13 --stop2=24 --skip1 --NNmode="final"; python database.py --env='+str(i)+' --start2=25 --stop2=33 --skip1 --NNmode="final"; python database.py --env='+str(i)+' --start2=34 --stop2=42 --skip1 --NNmode="final"; python database.py --env='+str(i)+' --start2=43 --stop2=51 --skip1 --NNmode="final"; python database.py --env='+str(i)+' --start2=52 --stop2=61 --skip1 --NNmode="final"; python database.py --env='+str(i)+' --start2=62 --skip1 --NNmode="final"; ')
    input("test")"""
    
    """with open(path_rai+'/final.txt', 'w') as f:
        for i in list(range(1,numEnv+1)) + EnvAppend2:
            f.write("python database.py --env="+str(i)+' --stop2=14 --NNmode="final"; python database.py --env='+str(i)+' --start2=15 --stop2=28 --skip1 --NNmode="final"; python database.py --env='+str(i)+' --start2=29 --stop2=38 --skip1 --NNmode="final"; python database.py --env='+str(i)+' --start2=39 --stop2=47 --skip1 --NNmode="final"; python database.py --env='+str(i)+' --start2=48 --stop2=60 --skip1 --NNmode="final"; python database.py --env='+str(i)+' --start2=62 --skip1 --NNmode="final"; ')
    input("test")"""

    """with open(path_rai+'/mixed3.txt', 'w') as f:
        for i in list(range(1,numEnv+1)) + EnvAppend:
            f.write("python database.py --env="+str(i)+' --stop2=40 --NNmode="mixed3"; python database.py --env='+str(i)+' --start2=41 --skip1 --NNmode="mixed3"; ')
    with open(path_rai+'/minimal.txt', 'w') as f:
         for i in range(1,100):
             f.write("python database.py --env="+str(i)+" --stop2=40; python database.py --env="+str(i)+" --start2=41 --skip1;\ \n")

    with open(path_rai+'/full.txt', 'w') as f:
         for i in range(1,100):
             f.write("python database.py --env="+str(i)+' --stop2=40 --NNmode="full"; python database.py --env='+str(i)+' --start2=41 --skip1 --NNmode="full";\ \n')

    with open(path_rai+'/dataset.txt', 'w') as f:
         for i in range(1,100):
             f.write("python database.py --env="+str(i)+' --stop2=40 --NNmode="dataset"; python database.py --env='+str(i)+' --start2=41 --skip1 --NNmode="dataset";\ \n')

    input("stop")"""

    if nenv==0:
        concatData(path_rai+'/dataset',start2,stop2, skip1=skip1,rand2=rand2, NNmode=NNmode, mixData=mixData, feasible=feasible)
    else:
        rai=rai_world.RaiWorld(path_rai, nenv, setup, "", verbose, NNmode=NNmode, datasetMode=dataMode, view=viewConfig)

        print("Finished loading. Train env "+str(nenv))
        if not skip1:
            dataSet(path_rai+'/dataset_new', rai, nenv, start1, stop1, mode=1)

        if not skip2:
            dataSet(path_rai+'/dataset_new', rai, nenv, start2, stop2, mode=2)
    

if __name__ == "__main__":
    main()