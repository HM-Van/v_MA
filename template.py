#dir_file="/home/my/rai-python/v_MA"
dir_file=os.path.abspath(os.path.dirname(__file__))
import sys
sys.path.append(dir_file+'/../ry/')
from libry import *

K=Config()
K.addFile(dir_file+'/rai-robotModels/pr2/pr2.g')
K.addFile(dir_file+'/models/Test_setup_'+str(102).zfill(3)+'.g')
#K.addFile(dir_file+'/models_final/Test_setup_'+str(1).zfill(3)+'.g')
#K.addFile(dir_file+'/rai-robotModels/baxter/baxter.g')

#K.addFile(dir_file+'/test/lgp-example.g')
#lgp=K.lgp(dir_file+"/models_final/fol-pickAndPlace.g")
lgp=K.lgp(dir_file+"/models/fol-pickAndPlace.g")

V=K.view()
lgp.walkToNode("(grasp pr2L red)")
lgp.optBound(BT.path, True)

komo = lgp.getKOMOforBound(BT.path)
#config = komo.getConfiguration(34)
#K.setFrameState(config)
K2=Config()
komo.getKFromKomo(K2, 14)
D=K2.view()
lgp2=K2.lgp(dir_file+"/models/fol-pickAndPlace.g")
lgp2.walkToDecision(5)
lgp2.optBound(BT.seq, True)




"""
lgp.walkToNode("(grasp pr2R stick) (push stickTip blue table2) (grasp pr2L red)")
#lgp.walkToNode("(grasp pr2R stick) (grasp pr2L red) (place pr2L red table2) (push stickTip red table2)")
lgp.walkToNode("(grasp baxterR stick) (grasp baxterL redBall)")
lgp.getDecisions()
lgp.optBound(BT.path, True)




dir_file="/home/my/rai-python/v_MA"
import sys
sys.path.append(dir_file+'/../ry/')
from libry import *
import rai_world

goalString="(on red green) (on green blue) (on blue yellow)"

rai=rai_world.RaiWorld(dir_file, 104, "minimal", goalString, 1, maxDepth=10, NNmode="mixed0", datasetMode=3, view=True)
"""

