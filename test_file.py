import sys
import os

dir_file=os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir_file+'/../ry/')
#print(sys.path)
from libry import *
import numpy as np

import rai_world
import time

def main():
    #dir_file=os.path.abspath(os.path.dirname(__file__))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rai_dir', type=str, default=dir_file)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--waitTime', type=float, default=0.01)


    parser.add_argument('--goal', type=str, default="(on green red) (on blue green)")
    parser.add_argument('--env', type=int, default=1)
    parser.add_argument('--setup', type=str, default="minimal")
    parser.add_argument('--skeleton', type=str, default="(grasp pr2R red) (place pr2R red green) (grasp pr2L green) (place pr2L green blue)")


    args = parser.parse_args()
    path_rai = args.rai_dir
    verbose=args.verbose
    waitTime= args.waitTime

    nenv=args.env
    goalString=args.goal
    setup=args.setup
    skeleton=args.skeleton

    #-------------------------------------------------------------------------------------------------------------------------
    """ model_dir="20191002-115531_full/"
    numLogicals=7

    dataInput=np.load(path_rai+'/dataset/'+model_dir+'Input.npy')
    dataInstruct=np.load(path_rai+'/dataset/'+model_dir+'Instruction.npy')
    dataLogicals=np.load(path_rai+'/dataset/'+model_dir+'Logicals.npy')

    graspIn=dataInput[dataInstruct[:,0]==1,:]
    graspAct=dataLogicals[dataInstruct[:,0]==1,:]

    graspIn1=np.concatenate((graspIn, np.zeros((graspAct.shape[0], 2*numLogicals))), axis=1)
    graspOut1=graspAct[:,numLogicals:2*numLogicals] #object
    graspIn2=np.concatenate((graspIn, np.zeros((graspAct.shape[0], numLogicals)), graspAct[:,numLogicals:2*numLogicals]), axis=1)
    graspOut2=graspAct[:,:numLogicals] #gripper

    graspFinalIn=np.zeros((2*graspIn.shape[0], graspIn.shape[1]+2*numLogicals))

    print(graspIn1.shape)
    graspFinalIn[0::2,:]=graspIn1
    graspFinalIn[1::2,:]=graspIn2

    graspFinalOut=np.zeros((2*graspIn.shape[0], numLogicals))
    graspFinalOut[0::2,:]=graspOut1
    graspFinalOut[1::2,:]=graspOut2


    placeIn=dataInput[dataInstruct[:,1]==1,:]
    placeAct=dataLogicals[dataInstruct[:,1]==1,:]

    placeIn1=np.concatenate((placeIn, np.zeros(placeAct.shape)), axis=1)
    placeOut1=placeAct[:,numLogicals:2*numLogicals] #object

    placeIn2=np.concatenate((placeIn, np.zeros((placeAct.shape[0], numLogicals)), placeAct[:,numLogicals:2*numLogicals], np.zeros((placeAct.shape[0], numLogicals))), axis=1)
    placeOut2=placeAct[:,:numLogicals] #gripper

    placeIn3=np.concatenate((placeIn, placeAct[:,0:2*numLogicals], np.zeros((placeAct.shape[0], numLogicals))), axis=1)
    placeOut3=placeAct[:,2*numLogicals:] #table

    placeFinalIn=np.zeros((3*placeIn.shape[0], placeIn.shape[1]+3*numLogicals))

    print(placeIn1.shape)
    placeFinalIn[0::3,:]=placeIn1
    placeFinalIn[1::3,:]=placeIn2
    placeFinalIn[2::3,:]=placeIn3

    placeFinalOut=np.zeros((3*placeIn.shape[0], numLogicals))
    placeFinalOut[0::3,:]=placeOut1
    placeFinalOut[1::3,:]=placeOut2
    placeFinalOut[2::3,:]=placeOut3

    graspNNIN=graspFinalIn.reshape(-1,2,graspIn.shape[1]+2*numLogicals)
    print("In")
    print(graspNNIN.shape)
    print(graspNNIN[0,:,:])

    print("Out")
    graspNNOut=graspFinalOut.reshape(-1,2,numLogicals)
    print(graspNNOut.shape)
    print(graspNNOut[0,:,:])

    print("-------------------")

    placeNNIN=placeFinalIn.reshape(-1,3,placeIn.shape[1]+3*numLogicals)
    print("In")
    print(placeNNIN.shape)
    print(placeNNIN[0,:,:])

    print("Out")
    placeNNOut=placeFinalOut.reshape(-1,3,numLogicals)
    print(placeNNOut.shape)
    print(placeNNOut[0,:,:])

    input("stop") """

    skeleton ="(grasp pr2L green) (place pr2L green blue) (grasp pr2R red) (place pr2R red green)"

    #-------------------------------------------------------------------------------------------------------------------------	
    print("Setting up basic Config and FOL for env: "+str(nenv))
    rai=rai_world.RaiWorld(path_rai, nenv, setup, goalString, verbose, maxDepth=20)
    actions=rai_world.splitStringPath(skeleton, verbose=0)

    #print(rai.K.getFrameNames())

    X0 = rai.K.getFrameState()
    """
    print(np.reshape(rai.encodeState(), (1,-1)))

    features= rai.encodeFeatures()

    print(features)
    print(rai.encodeInput(features))

    input("continue...")"""
    K=rai.K
    for act in actions:

        decisions=rai.lgp.getDecisions()
        input(decisions)
        rai.lgp.walkToRoot()
        rai.lgp.walkToNode(act,0)

        #rai.K.setFrameState(X0)

        try:
            komo1 = rai_world.runLGP(rai.lgp, BT.seq, verbose=0, view=False)
            komo2 = rai_world.runLGP(rai.lgp, BT.seqPath, verbose=0, view=True)
			
        except:
            print("Can not solve komo for skeleton:", skeleton)
            #rai.K.setFrameState(X0)
            break
        
        _, config = rai_world.applyKomo(komo2, rai.logicalNames, num=komo2.getPathFrames(rai.logicalNames).shape[0]-1, verbose=0)
        _, config2 = rai_world.applyKomo(komo1, rai.logicalNames, num=komo1.getPathFrames(rai.logicalNames).shape[0]-2, verbose=0)
        rai.K.setFrameState(config2,verb=0)
    
        #features2= rai.encodeFeatures()
        rai.lgp.nodeInfo()
        time.sleep(waitTime)
        input("check")
        #input("framestate for "+str(act))
        print("\n--------------------------\n")


    input("Press Enter to end Program...")

    """SKELETON:
    initial (pr2R) from 0 to 3
    initial (pr2L) from 0 to 3
    initial (red) from 0 to 3
    initial (green) from 0 to 3
    initial (blue) from 0 to 3
    initial (table1) from 0 to 3
    initial (table2) from 0 to 3

    
    touch (pr2R red) from 1 to 2
    stable (pr2R red) from 1 to 2
    touch (pr2L green) from 2 to 3
    stable (pr2L green) from 2 to 3
    above (red green) from 3 to 3
    stableOn (green red) from 3 to 3
    touch (pr2R red) from 3 to 3
    touch (red green) from 3 to 3
    SWITCHES:
    START  -->  stable (pr2R red) from 1 to 2
    START  -->  stable (pr2L green) from 2 to 3


    SKELETON:
    stable (pr2R red) from 0 to 1
    stable (pr2L green) from 1 to 2
    above (red green) from 2 to 2
    stableOn (green red) from 2 to 2
    touch (pr2R red) from 2 to 2
    touch (red green) from 2 to 2
    SWITCHES:
    START  -->  stable (pr2R red) from 0 to 1
    START  -->  stable (pr2L green) from 1 to 2
    stable (pr2R red) from 0 to 1  -->  stableOn (green red) from 2 to 2"""






if __name__ == "__main__":
    main()