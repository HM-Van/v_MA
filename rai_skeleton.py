import numpy as np
#import rai_world

import sys
import os
import time

dir_file=os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir_file+'/../ry/')
from libry import *

# File to obtain solutions from original LGP
# NOTE: goal formulations: (held object) (on table object)

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--rai_dir', type=str, default=dir_file)
	parser.add_argument('--env', type=int, default=1)
	parser.add_argument('--goal', type=str, default="(on red green) (on green blue)")
	
	args = parser.parse_args()
	path_rai = args.rai_dir
	nenv=args.env
	goalString=args.goal

	#-------------------------------------------------------------

	K=Config()
	K.addFile(path_rai+'/rai-robotModels/pr2/pr2.g')
	#K.addFile(path_rai+'/test/Test_setup_'+str(nenv).zfill(3)+'.g')
	K.addFile(path_rai+'/models/Test_setup_'+str(nenv).zfill(3)+'.g')

	lgp=K.lgp(path_rai+"/models/fol-pickAndPlace2.g")
	lgp.addTerminalRule(goalString)
	
	V=K.view()
	#input("test")

	starttime=time.time()
	try:
		lgp.run(0)
	except:
		endtime=time.time()
		print("crash after: "+str(endtime-starttime))

	endtime=time.time()
	print("Finished after: "+str(endtime-starttime))

	input("\nwait\n")

	#for command in CommandList:

	#	input("step " + command)
	#	lgp.walkToRoot()
	#	lgp.walkToNode(command)
	#	komo = t2.runLGP(lgp, BT.pathStep, verbose=0)
	#	envState, config = t2.applyKomo(komo, logicalNames, num=komo.getPathFrames(logicalNames).shape[1]-1, verbose=0)
	#	K2.setFrameState(config)

	#input("step path")
	#komo = t2.runLGP(lgp, BT.path, verbose=0)

	input("Press Enter to end Program...")



if __name__ == "__main__":
    main()
