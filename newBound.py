import os
dir_file=os.path.abspath(os.path.dirname(__file__))
import sys
sys.path.append(dir_file+'/../ry/')
from libry import *

import rai_world

def main():
    #dir_file=os.path.abspath(os.path.dirname(__file__))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rai_dir', type=str, default=dir_file)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--mode', type=int, default=1)

    args = parser.parse_args()
    path_rai = args.rai_dir
    mode= args.mode
    #-------------------------------------------------------------------------------------------------------------------------	
    K=Config()
    K.addFile(path_rai+'/rai-robotModels/pr2/pr2.g')
    K.addFile(path_rai+'/models/Test_setup_'+str(1).zfill(3)+'.g')
    lgp=K.lgp(path_rai+"/models/fol-pickAndPlace.g")

    V=K.view()
    skeleton="(grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green blue)"
    actions=rai_world.splitStringPath(skeleton, verbose=0, list_old=[])
    actions2=rai_world.splitStringStep(skeleton, verbose=0, list_old=[])

    baseName=[K.getFrameNames()[1]]


    if mode==1:
        print("pathStep Bound")
        for act in actions:
            lgp.walkToRoot()
            lgp.walkToNode(act,0)

            try:
                komo = rai_world.runLGP(lgp, BT.pathStep, verbose=0, view=True)
                
            except:
                print("Can not solve komo for skeleton:", skeleton)
                break
            
            config = komo.getConfiguration(komo.getPathFrames(baseName).shape[0]-1)
            K.setFrameState(config,verb=0)
        

            lgp.nodeInfo()
    else:
        print("test copy config")
        for act in actions2:
            lgp.walkToNode(act,0)

            try:
                komo = rai_world.runLGP(lgp, BT.path, verbose=0, view=True)
            except:
                print("Can not solve komo for skeleton:", skeleton)
                break
            
            K2=Config()
            komo.getKFromKomo(K2, komo.getPathFrames(baseName).shape[0]-1)
            K.copy(K2)
            lgp=K.lgp(path_rai+"/models/fol-pickAndPlace.g")
        

            lgp.nodeInfo()



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