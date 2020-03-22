import numpy as np
import rai_setup
import minimal_experiment as expert
import new_experiment0 as expertStack

import sys
import os
import random

dir_file=os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir_file+'/../ry/')
from libry import *


import time
import datetime

numLogicalType=3
numGoal=2
numGoalInstruct=2
numActInstruct=2

numSets=72
numEnv=99
EnvAppend=[28,29,30,31,37,38,39,40,46,47,48,49,55,56,57,58,64,65,66,67,73,74,75,76,82,83,84,85,91,92,93,94]
EnvAppend2=[28,37,46,55,64,73,82,91,29,38,47,56,65,74,83,92,30,39,48,57,66,75,84,93,31,40,49,58,67,76,85,94]

def encodeAction(commandString, grNames, objNames, tabNames):
    # encodes instruction and symbols in "classical" 2D way
    # gripper object table

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
		print("\nNot implemented")

	return instructionEn, logEn

def encodeAction3D(commandString, allNames): # !!!!! switches order: grasp object gripper, place object gripper table
    # encodes instruction and symbols in 3D waym in the order it gets trained
    # object gripper place
    instructionEn = np.zeros((1,numActInstruct))
    split_str = commandString.split(" ")

    if split_str[0] == '(grasp':
        instructionEn[(0,0)] = 1
        logEn = np.zeros((1, numLogicalType,len(allNames)))

        #object
        split_str[2]=split_str[2][:-1]
        logEn[(0, 0, allNames.index(split_str[2]))]=1

        #gripper
        logEn[(0, 1, allNames.index(split_str[1]))]=1

    elif split_str[0] == '(place':
        instructionEn[(0,1)] = 1
        logEn = np.zeros((1, numLogicalType,len(allNames)))

        #object
        split_str[2]=split_str[2]
        logEn[(0, 0, allNames.index(split_str[2]))]=1

        #gripper
        logEn[(0, 1, allNames.index(split_str[1]))]=1

        #table
        split_str[3]=split_str[3][:-1]
        logEn[(0, 2, allNames.index(split_str[3]))]=1


    else:
        logEn = np.zeros((1, 1))
        print("\nNot implemented")

    return instructionEn, logEn

def dataSet(path_dB, rai, nenv, start,stop,mode=2):
    komo=None

    K0=Config()
    K0.copy(rai.K)

    # Make dir if not exist
    if not(os.path.exists(path_dB)):
        os.makedirs(path_dB)

    if rai.NNmode =="dataset":
        if not(os.path.exists(path_dB+'/env'+str(nenv).zfill(3))):
            os.makedirs(path_dB+'/env'+str(nenv).zfill(3))
    else:
        if not(os.path.exists(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode)):
            os.makedirs(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode)

    # Start data set
    for nset in range(start,stop+1):
        # Get solutions
        if mode==2:
            if rai.NNmode in ["stack"]:
                solutions, goalString, numLoops = expertStack.getData(nset=nset, nenv=nenv-200)
            else:
                solutions, goalString, numLoops = expert.getData(nset=nset, nenv=nenv)
        elif mode==1:
            if rai.NNmode in ["stack"]:
                solutions, goalString, numLoops = expertStack.getData1(nset=nset, nenv=nenv-200)
            else:
                solutions, goalString, numLoops = expert.getData1(nset=nset, nenv=nenv)

        if numLoops==0:
            print("No solution for env "+str(nenv)+" set "+str(nset))
            continue
        
        # Check if file already exists
        if rai.NNmode in ["final", "stack"] and os.path.isfile(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode+'/set'+str(nset).zfill(mode+1)+'Input.npy'):
            print("Already trained env "+str(nenv)+" set "+str(nset))
            continue

        # Init input size
        if rai.NNmode in ["final", "stack"]:
            input_size = (numGoalInstruct + len(rai.objNames)+len(rai.tabNames))*numGoal + len(rai.logicalNames)*3
        else:
            NotImplementedError
        
        if rai.NNmode in ["stack"]:
            print(expertStack.getEnvInfo(nenv-200,"r"),expertStack.getEnvInfo(nenv-200,"g"),expertStack.getEnvInfo(nenv-200,"b"))

        # Init arrays for data set 
        inputArray = np.zeros((numLoops,input_size), dtype=float)
        instrArray = np.zeros((numLoops,numActInstruct), dtype=int)

        if rai.NNmode in ["final", "stack"]:
            logArray = np.zeros((numLoops,numLogicalType,len(rai.logicalNames)), dtype=int)

            
            feasibleArray = np.zeros((numLoops,2), dtype=int)  # act skeleton

            inputArray2 = np.zeros((numLoops,input_size), dtype=float)

            if rai.NNmode in ["final"]:
                prevInputArray = np.zeros((numLoops,4,input_size), dtype=float)
                prevInputArray2 = np.zeros((numLoops,4,input_size), dtype=float)
            elif rai.NNmode in ["stack"]:
                prevInputArray = np.zeros((numLoops,6,input_size), dtype=float)
                prevInputArray2 = np.zeros((numLoops,6,input_size), dtype=float)
        else:
            NotImplementedError


        i=0
        print("\n-------- Objective for set "+str(nset)+"/"+str(stop)+": "+goalString[0]+" --------")

        for sol in solutions:
            # Preprocess skeleton
            rai.K.copy(K0)
            commandList, commandStep=[], []
            commandList= rai_setup.splitStringPath(sol, list_old=[],verbose=0)
            commandStep= rai_setup.splitStringStep(sol, list_old=[],verbose=0)

            rai.lgp.walkToRoot()

            print("env "+str(nenv)+", set "+str(nset)+": "+sol)
            i_start=i
            solfeas=0

            j=0
            #init previous input for this solution
            
            if rai.NNmode in ["final"]:
                prevInput = np.zeros((1,4,input_size), dtype=float)
                prevInput2 = np.zeros((1,4,input_size), dtype=float)
            elif rai.NNmode in ["stack"]:
                prevInput = np.zeros((1,6,input_size), dtype=float)
                prevInput2 = np.zeros((1,6,input_size), dtype=float)

            else:
                NotImplementedError

            for command, clist in zip(commandStep,commandList):
                # For each high-level action
                print('\t'+str(i+1)+"/"+str(numLoops)+" "+command)

                # Encode state
                envState=rai.encodeState()

                for goal in goalString:
                    # For each sequence of goal formulations: encode objective                        
                    goalStep = rai_setup.splitStringStep(goal, list_old=[],verbose=0)
                    goalState= rai.encodeGoal(goalStep)

                    if rai.NNmode in ["final", "stack"]:
                            # Encode input
                            if rai.NNmode in ["final", "stack"]:
                                [inputArray[i,:], inputArray2[i,:]]=rai.encodeInput(envState, goalState=goalState)
                            else:
                                inputArray[i,:]=rai.encodeInput(envState, goalState=goalState)
                                NotImplementedError

                            # Add to previous input = Sequence of input
                            # Note: goalState was not consistent previously!! as prevInput gets overwritten
                            # previous entries have goalString[-1] as goalstate
                            # "final" dataset has to be fixed if implementation 3 is used
                            prevInput[0,j,:] = inputArray[i,:]
                            prevInput[0,:,:goalState.shape[1]]=goalState
                            #input(goalState.shape[1])
                            prevInputArray[i,:,:]=prevInput[0,:,:]
                            if rai.NNmode in ["final", "stack"]:
                                prevInput2[0,j,:] = inputArray2[i,:]
                                prevInput2[0,:,:goalState.shape[1]]=goalState
                                prevInputArray2[i,:,:]=prevInput2[0,:,:]

                            # Encode Output
                            try:
                                instrArray[i,:], logArray[i,:,:] = encodeAction3D(command,rai.logicalNames)
                                if j<(prevInput.shape[1]-1) and i%mode==mode-1:
                                    #input(prevInput.shape[1]-1)
                                    j=j+1

                            except:
                                print("failed!: "+command+"\t"+clist)

                            i=i+1
                    
                    #print(i)

                # Got to node in tree: partial skeleton
                rai.lgp.walkToRoot()
                try:
                    rai.lgp.walkToNode(clist,0)
                except:
                    break
                rai.K.copy(K0)
                if rai.NNmode in ["final", "stack"]:
                    # Solve LGP for seq/seqPath
                    komo = rai_setup.runLGP(rai.lgp, BT.seq, verbose=0, view=False)
                    komo = rai_setup.runLGP(rai.lgp, BT.seqPath, verbose=0, view=False)
                    if rai.lgp.returnFeasible(BT.seqPath):
                        feasibleArray[i-1,0]=1
                        if mode==2:
                            feasibleArray[i-2,0]=1

                        if clist==commandList[-1]:
                            solfeas=1

                komo.getKFromKomo(rai.K, komo.getPathFrames(rai.logicalNames).shape[0]-1)

            if rai.NNmode in ["final", "stack"]:
                # Set feasibility for skeleton
                feasibleArray[i_start:i,1]=solfeas
            
            #print(instrArray)
            #print(logArray)
            #print(inputArray)
            #print(inputArray2)
            #print(prevInputArray[i_start:i,:,:])
            #print(prevInputArray2)
            #input(feasibleArray)

            print("")

        if i==0:
            print("No feasible solution for env "+str(nenv)+" set "+str(nset))
            continue
        
        if rai.NNmode in ["final", "stack"]:
            print(str((np.where(~feasibleArray[:,1:2].any(axis=1))[0].shape[0])/mode)+" infeasible solutions")

        # Delete 0 rows
        inputArray=np.delete(inputArray, list(range(i, numLoops)), axis=0)
        instrArray=np.delete(instrArray, list(range(i, numLoops)), axis=0)
        logArray=np.delete(logArray, list(range(i, numLoops)), axis=0)

        if rai.NNmode in ["final", "stack"]:
            prevInputArray=np.delete(prevInputArray, list(range(i, numLoops)), axis=0)
            inputArray2=np.delete(inputArray, list(range(i, numLoops)), axis=0)
            prevInputArray2=np.delete(prevInputArray, list(range(i, numLoops)), axis=0)

        # Save data set
        if mode==2:
            tmpZero=3
        elif mode==1:
            tmpZero=2

        np.save(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode+'/set'+str(nset).zfill(tmpZero)+'Input',inputArray)
        np.save(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode+'/set'+str(nset).zfill(tmpZero)+'Instruction', instrArray)
        np.save(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode+'/set'+str(nset).zfill(tmpZero)+'Logicals', logArray)

        if rai.NNmode in ["final", "stack"]:
            np.save(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode+'/set'+str(nset).zfill(tmpZero)+'InputPrev', prevInputArray)
            np.save(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode+'/set'+str(nset).zfill(tmpZero)+'Feasible', feasibleArray)
            np.save(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode+'/set'+str(nset).zfill(tmpZero)+'Input2',inputArray2)
            np.save(path_dB+'/env'+str(nenv).zfill(3)+"_"+rai.NNmode+'/set'+str(nset).zfill(tmpZero)+'InputPrev2', prevInputArray2)

def offsetDataSet(Input, Logicals, prevInput, Input2, prevInput2, mode=2, NNmode="final"):
    # Init arrays
    finalInput=Input
    finalLogicals=Logicals
    finalprevInput=prevInput
    finalInput2=Input2
    finalprevInput2=prevInput2

    # Hard-coded: arrays of length of all logic types
    grNames=[0,1]
    objNames=[2,3,4]
    if NNmode=="stack":
        tabNames=[2,3,4,5]
        logicalNames=[0,1,2,3,4,5]
    elif NNmode=="final":
        tabNames=[2,3,4,5,6]
        logicalNames=[0,1,2,3,4,5,6]

    # Init input size: for NNmode=="final"
    input_size = (numGoalInstruct + len(objNames)+len(tabNames))*numGoal + len(logicalNames)*3
    
    # All possible permutations besides [0,1,2]
    if mode==2:
        switches=[[0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]

    else:
        switches=[[2,0,1], [1,2,0]]

    for switch in switches:
        # Permutations: obj in goal formulation, tables(3 obj) in goal formulation, obj in state
        Inputidx=(list(range(numGoalInstruct))+[x+numGoalInstruct for x in switch]+[x+numGoalInstruct+len(objNames) for x in switch]+list(range( numGoalInstruct + 2*len(objNames), 2*numGoalInstruct + len(objNames)+len(tabNames)))+ #goal formulation 1
                    [x+2*numGoalInstruct +len(objNames)+len(tabNames) for x in switch] + [x+2*numGoalInstruct +2*len(objNames)+len(tabNames) for x in switch] + # goal formulation 2
                    list(range(2*numGoalInstruct + 3*len(objNames)+len(tabNames), 2*(numGoalInstruct + len(objNames)+len(tabNames))+len(grNames)*3)) + # 2 grippers
                    [x+2*(numGoalInstruct + len(objNames) + len(tabNames))+3*(len(grNames) + switch[0]) for x in [0,1,2]]+ # obj switch[0]
                    [x+2*(numGoalInstruct + len(objNames) + len(tabNames))+3*(len(grNames) + switch[1]) for x in [0,1,2]]+ # obj switch[1]
                    [x+2*(numGoalInstruct + len(objNames) + len(tabNames))+3*(len(grNames) + switch[2]) for x in [0,1,2]]+ # obj switch[2]
                    list(range(2*(numGoalInstruct + len(objNames)+len(tabNames))+3*(len(grNames)+len(objNames)), input_size)) # table1 table2
                    )
        finalInput = np.concatenate((finalInput, Input[:, Inputidx]), axis=0)
        finalInput2= np.concatenate((finalInput2, Input2[:, Inputidx]), axis=0)
        finalprevInput =np.concatenate((finalprevInput,prevInput[:, :, Inputidx]), axis=0)
        finalprevInput2=np.concatenate((finalprevInput2,prevInput2[:, :, Inputidx]), axis=0)

        # Permutations: obj in state
        Logidx=list(range(len(grNames)))+[x + len(grNames) for x in switch]+list(range(len(grNames)+len(objNames), len(logicalNames)))
        finalLogicals = np.concatenate((finalLogicals, Logicals[:,:,Logidx]), axis=0)
    
    # duplicate if small number of samples
    if Input.shape[0]<=48 and mode==2:
        finalInput = np.concatenate((finalInput, finalInput), axis=0)
        finalInput2= np.concatenate((finalInput2, finalInput2), axis=0)
        finalprevInput =np.concatenate((finalprevInput,finalprevInput), axis=0)
        finalprevInput2=np.concatenate((finalprevInput2,finalprevInput2), axis=0)

        finalLogicals = np.concatenate((finalLogicals, finalLogicals), axis=0)


    return finalInput, finalLogicals, finalprevInput, finalInput2, finalprevInput2

def concatData(path_dB,start,stop, skip1=False, rand2=0, NNmode="minimal", mixData=False, exclude=False):

    obg = [[],[]]#[[15], [10,20,30,39,48,53,58,63,68]]
    or1 = [[2], [11,12,13,14,15,16,17,18,19,20]]
    og2 = [[],[]]#[[8], [3,13,23,33,42,59,60,61,62,63]]
    
    if exclude:
        # If exclude: initially all sets, excluded later
        arrSets=list(range(1,numSets+1))
        envSets=[5,7,9,11,15,17,21,23,26,30,31,36,37,41,45,47,52,53,55,61,62,67,69,70,74,76,79,85,86,87,92,98,99, 
                6,14,27,28,44,49,60,71,77,84,95]
                #34,43,54,63,72,81,90,96, 2, 4, 13, 25, 1, 18, 20]

    elif NNmode=="stack":
        #envSets=list(range(201,253))
        envSets= [i for i in range(201,253) if not i%4==0]
        arrSets=[29, 26,3, 31,4, 22,28, 6, 12,37, 18,35, 14]

    elif mixData:
        # Data set expansion
        arrSets=[6,7,21,27,32,62,68]#45 #67, 52 #71, 20 #12 61

        if rand2>len(arrSets):
            arrSets0=[x for x in list(range(1,numSets+1)) if x not in arrSets]
            random.shuffle(arrSets0)
            arrSets=arrSets+arrSets0[:rand2-len(arrSets)]
            input(arrSets)
        
        envSets=[5,7,9,11,15,17,21,23,26,30,31,36,37,41,45,47,52,53,55,61,62,67,69,70,74,76,79,85,86,87,92,98,99, 
                6,14,27,28,44,49,60,71,77,84,95]
    else:
        # Normal mode
        if rand2==0:
            #40
            arrSets=[46,59,36,11,13,40,57,19,20,35,65,28,23,42,25,66,30,53,18,7,39,44,24,31,16,56,71,21,45,49,4,22,38,29,10,68,62,27,9,70]#,17, 41, 54, 34, 26]
            #36
            #arrSets=[41,59,36,11,13,57,19,20,35,65,28,23,42,25,66,53,7,39,44,24,31,16,56,71,21,45,49,4,22,38,29,10,68,9,70,62]#,17, 41, 54, 34, 26, 18, 27, 30, 42, 40, 46] +41
        else:
            arrSets=np.arange(1,numSets+1)
            np.random.shuffle(arrSets)
            arrSets=arrSets[0:rand2]
        
        envSets=[5,7,9,11,15,21,23,26,30,31,36,37,41,45,47,52,53,55,61,62,67,69,70,74,76,79,85,86,87,92,98,99, 
                6,14,27,28,44,49,60,71,77,84,95,
                34,43,54,63,72,81,90,96]



    # Create folder name
    now=datetime.datetime.now()
    timestamp=str(now.year)+str(now.month).zfill(2)+str(now.day).zfill(2)+"-"+str(now.hour).zfill(2)+str(now.minute).zfill(2)+str(now.second).zfill(2)

    if NNmode in ["final", "stack"] and True:
        path_dB=path_dB+"_new"

    appendName="_"+NNmode

    # Lists for summary
    listEnv, listNoEnv=[],[]

    # Load first array
    for i in arrSets:
        if not os.path.isfile(path_dB+'/env'+str(envSets[0]).zfill(3)+appendName+'/set'+str(i).zfill(3)+'Input.npy'):
            continue

        start=i
        old1=np.load(path_dB+'/env'+str(envSets[0]).zfill(3)+appendName+'/set'+str(start).zfill(3)+'Input.npy')
        old2=np.load(path_dB+'/env'+str(envSets[0]).zfill(3)+appendName+'/set'+str(start).zfill(3)+'Instruction.npy')
        old3=np.load(path_dB+'/env'+str(envSets[0]).zfill(3)+appendName+'/set'+str(start).zfill(3)+'Logicals.npy')

        if NNmode in ["final", "stack"]:
            old6=np.load(path_dB+'/env'+str(envSets[0]).zfill(3)+appendName+'/set'+str(start).zfill(3)+'InputPrev.npy')
            old7=np.load(path_dB+'/env'+str(envSets[0]).zfill(3)+appendName+'/set'+str(start).zfill(3)+'Feasible.npy')
            old8=np.load(path_dB+'/env'+str(envSets[0]).zfill(3)+appendName+'/set'+str(start).zfill(3)+'Input2.npy')
            old9=np.load(path_dB+'/env'+str(envSets[0]).zfill(3)+appendName+'/set'+str(start).zfill(3)+'InputPrev2.npy')

            if mixData:
                old1, old3, old6, old8, old9 = offsetDataSet(old1, old3, old6, old8, old9, NNmode=NNmode)
                old2= np.tile(old2,(6,1))
                #old4= np.tile(old4,(6,1,1))
                old7= np.tile(old7,(6,1))
                if old2.shape[0]<=48*6:
                    # Duplicate if few samples
                    old2=np.concatenate((old2,old2), axis=0)
                    #old4=np.concatenate((old4,old4), axis=0)
                    old7=np.concatenate((old7,old7), axis=0)
            elif old2.shape[0]<=48:
                old1=np.concatenate((old1,old1), axis=0)
                old2=np.concatenate((old2,old2), axis=0)
                old3=np.concatenate((old3,old3), axis=0)
                old6=np.concatenate((old6,old6), axis=0)
                old7=np.concatenate((old7,old7), axis=0)
                old8=np.concatenate((old8,old8), axis=0)
                old9=np.concatenate((old9,old9), axis=0)
        break


    for nenv in envSets:
    # For all initial configurations
        #if nenv%4==0:
        #    continue

        # Check if init config exists
        if not(os.path.exists(path_dB+'/env'+str(nenv).zfill(3)+appendName)):
            print("skip env "+str(nenv))
            continue
        else:
            print("process env "+str(nenv))
        
        #print(expert.getEnvInfo(nenv,"r"),expert.getEnvInfo(nenv,"g"),expert.getEnvInfo(nenv,"b"))

        listSet, listNoSet=[],[]
        listSet.append(str(nenv).zfill(3))
        listNoSet.append(str(nenv).zfill(3))

        for nset in arrSets:
            # For all objectives
            if nenv is envSets[0] and nset is start and not nset in listSet:
                listSet.append(str(nset).zfill(3))
                continue
            
            # if data set expansion: with chance of 50%, chose objective that has full skeleton
            if mixData and random.random()>0.5 and NNmode in ["final"]:
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

            # If objective does not exist or is excluded
            if not os.path.isfile(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(3)+'Input.npy') or (exclude and nset in or1[1]+og2[1]+obg[1]):
                print("skip    env "+str(nenv)+" set "+str(nset).zfill(3))
                listNoSet.append(str(nset).zfill(3))
                continue

            #print("process env "+str(nenv)+" set "+str(nset).zfill(3))

            # Load new array
            new1=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(3)+'Input.npy')
            new2=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(3)+'Instruction.npy')
            new3=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(3)+'Logicals.npy')

            if NNmode in ["final", "stack"]:
                new6=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(3)+'InputPrev.npy')
                new7=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(3)+'Feasible.npy')
                new8=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(3)+'Input2.npy')
                new9=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(3)+'InputPrev2.npy')

                if mixData:
                    new1, new3, new6, new8, new9 = offsetDataSet(new1, new3, new6, new8, new9, NNmode=NNmode)
                    new2= np.tile(new2,(6,1))
                    #new4= np.tile(new4,(6,1,1))
                    new7= np.tile(new7,(6,1))
                    #print(new2.shape)
                    if new2.shape[0]<=48*6:
                        # Duplicate if few samples
                        new2=np.concatenate((new2,new2), axis=0)
                        #new4=np.concatenate((new4,new4), axis=0)
                        new7=np.concatenate((new7,new7), axis=0)
                elif new2.shape[0]<=48:
                    new1=np.concatenate((new1,new1), axis=0)
                    new2=np.concatenate((new2,new2), axis=0)
                    new3=np.concatenate((new3,new3), axis=0)
                    new6=np.concatenate((new6,new6), axis=0)
                    new7=np.concatenate((new7,new7), axis=0)
                    new8=np.concatenate((new8,new8), axis=0)
                    new9=np.concatenate((new9,new9), axis=0)

                if NNmode in ["final", "stack"]:
                    # Skip if for some reason dimensions do not match
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

                # Concatenate arrays
                old8=np.concatenate((old8,new8), axis=0)
                old9=np.concatenate((old9,new9), axis=0)

                old7=np.concatenate((old7,new7), axis=0)

                old6=np.concatenate((old6,new6), axis=0)

            old1=np.concatenate((old1,new1), axis=0)
            old2=np.concatenate((old2,new2), axis=0)
            old3=np.concatenate((old3,new3), axis=0)

            if not str(nset).zfill(3) in listSet:
                listSet.append(str(nset).zfill(3))
        
        if not skip1:
            listSet.append("|")
            if NNmode in ["stack"]:
                set1=range(1,13)
            
            elif mixData:
                # randomly select one objects
                set1_tmp=[[1,2,3,4,15],[6,7,8,10,5],[11,12,13,14,9]]
                set1 = set1_tmp[random.randint(0,2)]
            else:
                set1=range(1,16)

            for nset in set1:
                # If objective does not exist or excluded
                if not os.path.isfile(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(2)+'Input.npy') or (exclude and nset in or1[0]+og2[0]+obg[0]):
                    print("skip    env "+str(nenv)+" set "+str(nset).zfill(2))
                    listNoSet.append(str(nset).zfill(2))
                    continue

                #print("process env "+str(nenv)+" set "+str(nset).zfill(2))

                #Load new arrays
                new1=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(2)+'Input.npy')
                new2=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(2)+'Instruction.npy')
                new3=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(2)+'Logicals.npy')

                if NNmode in ["final", "stack"]:
                    new6=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(2)+'InputPrev.npy')
                    new7=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(2)+'Feasible.npy')
                    
                    new8=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(2)+'Input2.npy')
                    new9=np.load(path_dB+'/env'+str(nenv).zfill(3)+appendName+'/set'+str(nset).zfill(2)+'InputPrev2.npy')

                    if mixData:
                        new1, new3, new6, new8, new9 = offsetDataSet(new1, new3, new6, new8, new9, mode=1, NNmode=NNmode)
                        new2= np.tile(new2,(3,1))
                        #new4= np.tile(new4,(3,1,1))
                        new7= np.tile(new7,(3,1))

                    if NNmode in ["final", "stack"]:
                        # Skip if for some reason dimensions do not match
                        try:
                            assert new1.shape[0] == new2.shape[0], "Inconsistent dimensions InstructionPrev"+str(new1.shape[0])+" != "+str(new2.shape[0])
                            assert new1.shape[0] == new3.shape[0], "Inconsistent dimensions LogicalsPrev"+str(new1.shape[0])+" != "+str(new3.shape[0])
                            assert new1.shape[0] == new6.shape[0], "Inconsistent dimensions InputPrev"+str(new1.shape[0])+" != "+str(new6.shape[0])
                            assert new1.shape[0] == new7.shape[0], "Inconsistent dimensions Feasible"+str(new1.shape[0])+" != "+str(new7.shape[0])
                
                            assert new1.shape[0] == new8.shape[0], "Inconsistent dimensions Input2"+str(new1.shape[0])+" != "+str(new8.shape[0])
                            assert new1.shape[0] == new9.shape[0], "Inconsistent dimensions InputPrev2"+str(new1.shape[0])+" != "+str(new9.shape[0])
                        except:
                            print("skip    env "+str(nenv)+" set "+str(nset).zfill(2)+" as inconsistent")
                            listNoSet.append(str(nset).zfill(2))
                            continue

                    # Concatenate arrays
                    old8=np.concatenate((old8,new8), axis=0)
                    old9=np.concatenate((old9,new9), axis=0)
                        
                    old7=np.concatenate((old7,new7), axis=0)

                    old6=np.concatenate((old6,new6), axis=0)

                old1=np.concatenate((old1,new1), axis=0)
                old2=np.concatenate((old2,new2), axis=0)
                old3=np.concatenate((old3,new3), axis=0)
                listSet.append(str(nset).zfill(2))
            
        listEnv.append(listSet)
        listNoEnv.append(listNoSet)
        print("")


    if NNmode in ["final", "stack"]:
        # Get feasible and infeasible samples

        infeasible=np.where(~old7[:,1:2].any(axis=1))[0]
        feasible=np.where(old7[:,1:2].any(axis=1))[0]
        new1=old1 #input
        new2=old2 #instr
        new3=old3 #log
        new6=old6 #inputprev
        new7=old7 #feas

        old1=old1[feasible,:]
        old2=old2[feasible,:]
        old3=old3[feasible,:,:]
        old6=old6[feasible,:,:]
        old7=old7[feasible,:]

        new1=new1[infeasible,:] #input
        new2=new2[infeasible,:] #instr
        new3=new3[infeasible,:,:] #log
        new6=new6[infeasible,:,:] #inputprev
        new7=new7[infeasible,:] #infeas

        if NNmode in ["final", "stack"]:
            new8=old8
            new9=old9
            old8=old8[feasible,:]
            new8=new8[infeasible,:]
            old9=old9[feasible,:,:]
            new9=new9[infeasible,:,:]

    # Last check fpr dimensions
    assert old1.shape[0] == old2.shape[0], "Inconsistent dimensions Instruction: "+str(old1.shape)[0]+" != "+str(old2.shape[0])
    assert old1.shape[0] == old3.shape[0], "Inconsistent dimensions Logicals"+str(old1.shape)[0]+" != "+str(old3.shape[0])

    if NNmode in ["final", "stack"]:
        assert old1.shape[0] == old6.shape[0], "Inconsistent dimensions InputPrev"+str(old1.shape)[0]+" != "+str(old6.shape[0])
        assert old1.shape[0] == old7.shape[0], "Inconsistent dimensions Feasible"+str(old1.shape)[0]+" != "+str(old7.shape[0])
        
        assert new1.shape[0] == new7.shape[0], "Inconsistent dimensions Infeasible"+str(new1.shape)[0]+" != "+str(old7.shape[0])
        assert new1.shape[0] == new2.shape[0], "Inconsistent dimensions InfeasibleInstr"+str(new1.shape)[0]+" != "+str(new2.shape[0])
        assert new1.shape[0] == new3.shape[0], "Inconsistent dimensions InfeasibleLog"+str(new1.shape)[0]+" != "+str(new3.shape[0])
        assert new1.shape[0] == new6.shape[0], "Inconsistent dimensions InfeasibleInputPrev"+str(old1.shape)[0]+" != "+str(new6.shape[0])
        
        assert old1.shape[0] == old8.shape[0], "Inconsistent dimensions Input2"+str(old1.shape)[0]+" != "+str(old8.shape[0])
        assert new1.shape[0] == new8.shape[0], "Inconsistent dimensions InfeasibleInput2"+str(new1.shape)[0]+" != "+str(old8.shape[0])
        assert old1.shape[0] == old9.shape[0], "Inconsistent dimensions InputPrev2"+str(old1.shape)[0]+" != "+str(old9.shape[0])
        assert new1.shape[0] == new9.shape[0], "Inconsistent dimensions InfeasibleInputPrev2"+str(new1.shape)[0]+" != "+str(old9.shape[0])

    # Create Results directory
    if not(os.path.exists(path_dB+'/'+timestamp+appendName)):
        os.makedirs(path_dB+'/'+timestamp+appendName)

    # Save arrays
    np.save(path_dB+'/'+timestamp+appendName+'/Input',old1)
    np.save(path_dB+'/'+timestamp+appendName+'/Instruction', old2)
    np.save(path_dB+'/'+timestamp+appendName+'/Logicals', old3)

    if NNmode in ["final", "stack"]:
        np.save(path_dB+'/'+timestamp+appendName+'/Feasible', old7)
        np.save(path_dB+'/'+timestamp+appendName+'/InputInfeasible', new1)
        np.save(path_dB+'/'+timestamp+appendName+'/InFeasible', new7)
        np.save(path_dB+'/'+timestamp+appendName+'/InFeasibleInstr', new2)
        np.save(path_dB+'/'+timestamp+appendName+'/InFeasibleLog', new3)
        np.save(path_dB+'/'+timestamp+appendName+'/InFeasibleInputPrev', new6)

        np.save(path_dB+'/'+timestamp+appendName+'/Input_feat',old8)
        np.save(path_dB+'/'+timestamp+appendName+'/InputInfeasible_feat', new8)
        np.save(path_dB+'/'+timestamp+appendName+'/InFeasibleInputPrev_feat', new9)

        if old1.shape[0]>80000:
            idxSplit=int(old1.shape[0]/3)
            
            np.save(path_dB+'/'+timestamp+appendName+'/InputPrev1', old6[:idxSplit,:,:])
            np.save(path_dB+'/'+timestamp+appendName+'/InputPrev2', old6[idxSplit:2*idxSplit,:,:])
            np.save(path_dB+'/'+timestamp+appendName+'/InputPrev3', old6[2*idxSplit:,:,:])
            if NNmode in ["final", "stack"]:
                np.save(path_dB+'/'+timestamp+appendName+'/InputPrev1_feat', old9[:idxSplit,:,:])
                np.save(path_dB+'/'+timestamp+appendName+'/InputPrev2_feat', old9[idxSplit:2*idxSplit,:,:])
                np.save(path_dB+'/'+timestamp+appendName+'/InputPrev3_feat', old9[2*idxSplit:,:,:])

        elif old1.shape[0]>80000:
            idxSplit=int(old1.shape[0]/2)

            np.save(path_dB+'/'+timestamp+appendName+'/InputPrev1', old6[:idxSplit,:,:])
            np.save(path_dB+'/'+timestamp+appendName+'/InputPrev2', old6[idxSplit:,:,:])
            if NNmode in ["final", "stack"]:
                np.save(path_dB+'/'+timestamp+appendName+'/InputPrev1_feat', old9[:idxSplit,:,:])
                np.save(path_dB+'/'+timestamp+appendName+'/InputPrev2_feat', old9[idxSplit:,:,:])
        else:

            np.save(path_dB+'/'+timestamp+appendName+'/InputPrev', old6)
            if NNmode in ["final", "stack"]:
                np.save(path_dB+'/'+timestamp+appendName+'/InputPrev_feat', old9)

    # Create Summary
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

    if NNmode in ["mixed3", "mixed2", "final", "stack"]:
        tmpstr=" feasible and "+str(new1.shape[0])+" infeasible"
    else:
        tmpstr=""
    print("---Finished dataset with "+str(old1.shape[0])+tmpstr+" samples---")

    with open(path_dB+'/'+timestamp+appendName+'/Summary.txt', 'w') as f:
        f.write("Dataset with "+str(old1.shape[0])+tmpstr+" samples\n\n")
        f.write("Environments:\n\t")
        for i in envSets:
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
    parser.add_argument('--NNmode', type=str, default="final")
    parser.add_argument('--datasetMode', type=int, default=1)
    parser.add_argument('--viewConfig', dest='viewConfig', action='store_true')
    parser.set_defaults(viewConfig=False)

    parser.add_argument('--mixData', dest='mixData', action='store_true')
    parser.set_defaults(mixData=False)

    parser.add_argument('--rand2', type=int, default=0)

    parser.add_argument('--skip1', dest='skip1', action='store_true')
    parser.set_defaults(skip1=False)
    parser.add_argument('--skip2', dest='skip2', action='store_true')
    parser.set_defaults(skip2=False)

    parser.add_argument('--exclude', dest='exclude', action='store_true')
    parser.set_defaults(exclude=False)

    parser.add_argument('--model_dir', type=str, default='')
    
    args = parser.parse_args()
    path_rai = args.rai_dir
    verbose=args.verbose
    start2=args.start2 #first set for 2 goal formulations to simulate
    stop2=args.stop2 #last set for 2 goal formulations to simulate
    start1=args.start1 #first set for 1 goal formulation to simulate
    stop1=args.stop1 #last set for 1 goal formulation to simulate
    nenv=args.env
    setup=args.setup
    NNmode=args.NNmode
    dataMode=args.datasetMode # 1 5(global coord) 2 6(relative coord) 3 7(global coord+encoder) 4 8(relative coord+encoder) # 1-4 initial setup 1-5 modified setup
    exclude=args.exclude #Excludes hardcoded objectives when assebling data set

    skip1=args.skip1 #skip objectives consisting of 1 goal formulation
    skip2=args.skip2 #skip objectives consisting of 2 goal formulations
    rand2=args.rand2 #random number of objectives consisting of 2 goal formulations for data set
    viewConfig=args.viewConfig
    mixData=args.mixData # does data set expansion

    #with open(path_rai+'/stack.txt', 'w') as f:
    #    for i in range(201,253):
    #        f.write("python database.py --env="+str(i)+' --stop1=12 --NNmode="stack" --skip2; python database.py --env='+str(i)+' --stop2=8 --skip1 --NNmode="stack"; python database.py --env='+str(i)+' --start2=9 --stop2=14 --skip1 --NNmode="stack"; python database.py --env='+str(i)+' --start2=15 --stop2=20 --skip1 --NNmode="stack"; python database.py --env='+str(i)+' --start2=21 --stop2=26 --skip1 --NNmode="stack"; python database.py --env='+str(i)+' --start2=27 --stop2=31 --skip1 --NNmode="stack"; python database.py --env='+str(i)+' --start2=32 --stop2=36 --skip1 --NNmode="stack"; python database.py --env='+str(i)+' --start2=37 --stop2=42 --skip1 --NNmode="stack"; ')
    #input("test")
    
    """with open(path_rai+'/final2.txt', 'w') as f:
        for i in [28, 37, 46, 55, 64, 73, 82, 91, 29, 38, 47, 56, 65, 74, 83, 92, 30, 39, 48, 57, 66, 75, 84, 93, 31, 40, 49, 58, 67, 76, 85, 94, 32, 41, 50, 59, 68, 77, 86, 95, 33, 42, 51, 60, 69, 78, 87, 96, 34, 43, 52, 61, 70, 79, 88, 97, 35, 44, 53, 62, 71, 80, 89, 98, 36, 45, 54, 63, 72, 81, 90, 99]:
            f.write("python database.py --env="+str(i)+' --stop2=12 --NNmode="final"; python database.py --env='+str(i)+' --start2=13 --stop2=24 --skip1 --NNmode="final"; python database.py --env='+str(i)+' --start2=25 --stop2=33 --skip1 --NNmode="final"; python database.py --env='+str(i)+' --start2=34 --stop2=42 --skip1 --NNmode="final"; python database.py --env='+str(i)+' --start2=43 --stop2=51 --skip1 --NNmode="final"; python database.py --env='+str(i)+' --start2=52 --stop2=61 --skip1 --NNmode="final"; python database.py --env='+str(i)+' --start2=62 --skip1 --NNmode="final"; ')
    input("test")"""
    
    """with open(path_rai+'/final.txt', 'w') as f:
        for i in list(range(1,numEnv+1)) + EnvAppend2:
            f.write("python database.py --env="+str(i)+' --stop2=14 --NNmode="final"; python database.py --env='+str(i)+' --start2=15 --stop2=28 --skip1 --NNmode="final"; python database.py --env='+str(i)+' --start2=29 --stop2=38 --skip1 --NNmode="final"; python database.py --env='+str(i)+' --start2=39 --stop2=47 --skip1 --NNmode="final"; python database.py --env='+str(i)+' --start2=48 --stop2=60 --skip1 --NNmode="final"; python database.py --env='+str(i)+' --start2=62 --skip1 --NNmode="final"; ')
    input("test")"""

    if nenv==0:
        # Create data set
        concatData(path_rai+'/dataset',start2,stop2, skip1=skip1,rand2=rand2, NNmode=NNmode, mixData=mixData, exclude=exclude)
    else:
        # Get training samples

        rai=rai_setup.RaiWorld(path_rai, nenv, setup, "", verbose, NNmode=NNmode, datasetMode=dataMode, view=viewConfig)

        print("Finished loading. Train env "+str(nenv))
        if not skip1:
            dataSet(path_rai+'/dataset_new', rai, nenv, start1, stop1, mode=1)

        if not skip2:
            dataSet(path_rai+'/dataset_new', rai, nenv, start2, stop2, mode=2)
    

if __name__ == "__main__":
    main()