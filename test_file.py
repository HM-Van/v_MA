import sys
import os

dir_file=os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir_file+'/../ry/')
#print(sys.path)
from libry import *
import numpy as np

import rai_world
import time

# Old file for testing of new bound -> see newBound.py

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
    for act in actions:

        decisions=rai.lgp.getDecisions()
        input(decisions)
        rai.lgp.walkToRoot()
        rai.lgp.walkToNode(act,0)

        rai.K.setFrameState(X0)

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