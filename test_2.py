#!/usr/bin/python3

# python test_1.py --rai_dir ~/rai-python
import sys
import os

dir_file=os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir_file+'/../ry/')
from libry import *


#from numpy import *

import rai_policy as NN
import tensorflow as tf

import rai_world
import minimal_experiment

import datetime
import time
import shutil

# Not used anymore -> see main.py

#Notes
	#K2.getFrameState() #get 7d state(3pos 4rot) for all frames
	
	## lgp functions, work with nodes
	# lgp.getDecisions()
	# lgp.walkToDecision(uint) #uint arg: index, returns: decision in get Decisions of this index
	# lgp.walkToRoot()
	# lgp.walkToParent()
	# lgp.walkToNode() #const char*: whole decision path from root to node
	# lgp.viewTree() # currently error

	#lgp.optBound(BT.pose, True); #BT.seq (just key frames), BT.path (the full path) #BT.seqPath (seeded with the BT.seq result)
	#lgp.nodeInfo() # constraints, costs, feasible, decision, path, state(symbolic)

	##Show all possible decisions
	#decisions=lgp.getDecisions()
	#print(len(decisions),"Decisions possible:")
	#print(decisions)

	## for goal on x y: x type object, y type table
	#lgp.addTerminalRule("(on red blue) (on green red) (on blue table1)")
	#lgp.run(2) # runs endlessly. How to stop?

def findFeasible(rai_net, envState, action, Decisions=None, feasThresh=0.4, calcProb=True):
	if calcProb:
		_, feasSke, _, _ = rai_net.predictFeas(envState, rai_net.encodeAction(action))
		prob=0
	else:
		feasSke=0

	if feasSke<feasThresh:
		idxBest, Decisions=rai_net.evalFeasPredictions(envState, Decisions)
		action=Decisions[0][idxBest]
		prob=Decisions[1][idxBest]
		#print(Decisions)

	return action, prob


def findTerminal(lgp):
	decisions=lgp.getDecisions()
	found= False
	decision=""

	for i in range(len(decisions)):
		lgp.walkToDecision(i)
		if lgp.isTerminal():
			lgp.walkToParent()
			found = True
			decision = decisions[i]
			break
		lgp.walkToParent()

	return found, decision

def createResults(rai_net, cheat_terminal = False, cheat_goalstate=False,cheat_tree=False,cheat_feas=False, start=1, feasWeight=None, feasThresh=None, planOnly=False, test=False):
	if rai_net.setup == "minimal":
		append_folder=""
		if cheat_terminal:
			append_folder=append_folder+"_terminal"
		if cheat_goalstate:
			append_folder=append_folder+"_goalstate"
		if cheat_tree:
			append_folder=append_folder+"_tree"
		if cheat_feas:
			append_folder=append_folder+"_feas"
			if not feasThresh==None:
				append_folder=append_folder+str(int(feasThresh*100))
			if not feasWeight==None:
				append_folder=append_folder+str(int(feasWeight*100))
		
		path=rai_net.path_rai+'/result_minimal/'+rai_net.model_dir+append_folder
		if not(os.path.exists(path)):
				os.makedirs(path)
		
		if start==1:
			if rai_net.dataMode in [11,12,13,14] and rai_net.NNmode in ["mixed10"]:
				appendmode="mixed"+str(rai_net.dataMode)
			elif rai_net.dataMode in [11,12,13,14] and rai_net.NNmode in ["FFnew", "minimal"]:
				appendmode=rai_net.NNmode+str(rai_net.dataMode)
			else:
				appendmode=rai_net.NNmode

			if planOnly:
				append_test="_plan"
			else:
				append_test=""
			if test:
				append_test="_test"

			#if os.path.exists(path+'/env'+str(rai_net.nenv).zfill(3)+'.txt'):
			#	os.remove(path+'/env'+str(rai_net.nenv).zfill(3)+'.txt') #this deletes the file
			#input(path+'/env'+str(rai_net.nenv).zfill(3)+'.txt')
			shutil.copyfile(rai_net.path_rai+'/logs/'+rai_net.model_dir+"_"+appendmode+'/params.txt',path+'/params.txt')
			with open(path+'/env'+str(rai_net.nenv).zfill(3)+append_test+'.txt', 'a+') as f:
				f.write("\n\n-----Result for env "+str(rai_net.nenv)+"-----\n\n")

		return path

def writeResults(rai_net,skeleton,typeDecision,successmsg,path, goalnumber_string="", planOnly=False, feasible=True, tries=0, test=False):
	
	if rai_net.setup == "minimal":
		if planOnly:
			append_test="_plan"
		else:
			append_test=""
		if test:
				append_test="_test"

		
		if feasible:
			strfeas=" feasible 1"
		else:
			strfeas=" infeasible 0"

		with open(path+'/env'+str(rai_net.nenv).zfill(3)+append_test+'.txt', 'a+') as f:
			f.write("----"+successmsg+" "+goalnumber_string+": "+rai_net.goalString_orig+strfeas+"----\n")
			if tries>0 and feasible:
				f.write("---- in "+str(tries)+" tries ----\n")

			actions=rai_world.splitStringStep(skeleton, list_old=[],verbose=0)
			for act, typedec in zip(actions,typeDecision):
				f.write("\t\t  "+typedec[0]+": "+act)
				if typedec[0]=="NN":
					if rai_net.NNmode in "mixed3" or rai_net.dataMode==3:
						f.write("[ "+str(round(typedec[1],3))+" "+str(round(typedec[2],3))+" "+str(round(typedec[3],3))+" "+str(round(typedec[4],3))+" "+str(round(typedec[5],3))+" ]")
					elif rai_net.dataMode in [11,12,13,14]:
						f.write("[ "+str(typedec[1])+" "+str(typedec[2])+" ]")
						if not(rai_net.feas_net.modelFeasible is None and rai_net.feas2_net.modelFeasible is None):
							f.write("[ "+str(round(typedec[3],3))+" "+str(round(typedec[4],3))+" "+str(round(typedec[5],3))+" "+str(round(typedec[6],3))+" "+str(round(typedec[7],3))+" "+str(round(typedec[8],3))+" ]")
					f.write("\n")
				elif typedec[0]=="MP":
					if rai_net.NNmode in "mixed3" or rai_net.dataMode==3:
						f.write("[ "+str(round(typedec[3],3))+" "+str(round(typedec[4],3))+" "+str(round(typedec[5],3))+" "+str(round(typedec[6],3))+" "+str(round(typedec[7],3))+" ]")
					elif rai_net.dataMode in [11,12,13,14]:
						f.write("[ "+str(typedec[3])+" "+str(typedec[4])+" ]")
						if not(rai_net.feas_net.modelFeasible is None and rai_net.feas2_net.modelFeasible is None):
							f.write("[ "+str(round(typedec[5],3))+" "+str(round(typedec[6],3))+" "+str(round(typedec[7],3))+" "+str(round(typedec[8],3))+" "+str(round(typedec[9],3))+" "+str(round(typedec[10],3))+" ]")
					f.write("\twith prob "+str(round(typedec[2],3))+" instead of "+typedec[1]+"\n")
				elif typedec[0]=="FD":
					if rai_net.NNmode in "mixed3" or rai_net.dataMode==3:
						f.write("[ "+str(round(typedec[3],3))+" "+str(round(typedec[4],3))+" "+str(round(typedec[5],3))+" "+str(round(typedec[6],3))+" "+str(round(typedec[7],3))+" ]")
					f.write("\twith prob "+str(round(typedec[2],3))+" instead of "+typedec[1]+"\n")
				else:
					NotImplementedError
			f.write("--------------------\n\n")
			
		
	
def printResult(rai_net, skeleton):
	print("\n----Final Result----")
	print("Goal\t\t: "+rai_net.goalString_orig)
	print("Final skeleton\t: ")

	actions=rai_world.splitStringStep(skeleton, list_old=[],verbose=0)
	for act in actions:
		print("\t\t  "+act)
	print("--------------------")
	print("")

def buildSkeleton(rai_net, cheat_terminal = False, cheat_goalstate=False,cheat_tree=False, showFinal=True, waitTime=0, cheat_feas=False, feasThresh=0.4, planOnly=False,
					infeasibleSkeletons=[]):

	skeleton=""
	typeDecision=[]
	tmpDes=[]
	idxtmpdes=0
	decisions=None
	X0 = rai_net.K.getFrameState()

	if not infeasibleSkeletons==[]:
		print("Next try")

	envState=rai_net.encodeState()
	if cheat_goalstate or rai_net.numGoal_orig>2:
			goals_tmp, unfullfilled, change=rai_net.preprocessGoalState(cheatGoalState=cheat_goalstate)
			inputState=rai_net.encodeInput(envState, goalState=goals_tmp)
			if change and rai_net.setup=="minimal":
				print("\tGoal changed to "+unfullfilled[0])
	else:
		
		inputState=rai_net.encodeInput(envState)

	#lgp.viewTree()

	successmsg="Failed to reach goal"

	if not rai_net.lgp.isTerminal() and not len(rai_net.preprocessGoals())==0:
		outEncoded = rai_net.processPrediction(inputState)
		outDecoded = rai_net.decodeAction1(outEncoded)
		#print(outDecoded)

		if cheat_feas and (rai_net.dataMode in [2,3]):
			old=outDecoded

			if (outDecoded not in rai_net.lgp.getDecisions() and not rai_net.lgp.getDecisions() == []):
				calcProb=False
			else:
				calcProb=True

			outDecoded, prob = findFeasible(rai_net, envState, outDecoded, Decisions=decisions, feasThresh=feasThresh, calcProb=calcProb)
			if old==outDecoded:
				print("NN Decision", outDecoded)
				tmpDes.append("NN")
				idxtmpdes=1

			else:
				print("New Feas Decision: ", outDecoded, "\twith probability", prob)
				print("\tInstead of: ", old)
				#tmpDes=[]
				tmpDes.append("FD")
				tmpDes.append(old)
				tmpDes.append(prob)
				idxtmpdes=3

		elif outDecoded in infeasibleSkeletons or (outDecoded not in rai_net.lgp.getDecisions() and not rai_net.lgp.getDecisions() == []) or cheat_tree or rai_net.checkPlaceSame(outDecoded):
			old=outDecoded
			if not outDecoded in infeasibleSkeletons:
				outDecoded, prob = rai_net.evalPredictions(inputState, prevSke=outDecoded)
			else:
				outDecoded, prob = rai_net.evalPredictions(inputState, infeasible=outDecoded, prevSke=outDecoded)
			print("New Decision: ", outDecoded, "\twith probability", prob)
			print("\tInstead of: ", old)
			#tmpDes=[]
			tmpDes.append("MP")
			tmpDes.append(old)
			tmpDes.append(prob)
			idxtmpdes=3
			#typeDecision.append(tmpDes)
		else:
			print("NN Decision", outDecoded)
			#tmpDes=[]
			tmpDes.append("NN")
			idxtmpdes=1
			#typeDecision.append(tmpDes)
		skeleton = outDecoded
		depth=1
	
	else:
		return skeleton, typeDecision, "Successfully reached goal", True

	while True:

		rai_net.lgp.walkToRoot()
		rai_net.lgp.walkToNode(skeleton,0)

		try:
			if rai_net.dataMode in [11,12,13,14]:
				komo = rai_world.runLGP(rai_net.lgp, BT.seq, verbose=0, view=False)
				if not planOnly:
					komo = rai_world.runLGP(rai_net.lgp, BT.seqPath, verbose=0, view=False)
			else:
				komo = rai_world.runLGP(rai_net.lgp, BT.path, verbose=0, view=False)
		except:
			print("Can not solve komo path for skeleton:", skeleton)
			successmsg="KOMO path failed for goal"
			rai_net.lgp.walkToRoot()
			rai_net.K.setFrameState(X0, verb=0)
			break

		if rai_net.NNmode in ["mixed3", "mixed2"] or rai_net.dataMode in [3,2]:
			feasAct, feasSke, _, feasAct1, feasSke1,_ = rai_net.predictFeas(envState, rai_net.encodeAction(outDecoded))
			if rai_net.lgp.returnFeasible(BT.path):
				tmpDes.append(1)
			else:
				tmpDes.append(0)
			tmpDes.append(feasAct)
			tmpDes.append(feasSke)
			tmpDes.append(feasAct1)
			tmpDes.append(feasSke1)
			#print(tmpDes, idxtmpdes)
			print("\t"+str(tmpDes[idxtmpdes])+" "+str(tmpDes[idxtmpdes+1])+" "+str(tmpDes[idxtmpdes+2])+" "+str(tmpDes[idxtmpdes+3])+" "+str(tmpDes[idxtmpdes+4]))
		elif rai_net.dataMode in [11,12,13,14]:
			if rai_net.lgp.returnFeasible(BT.seq):
				tmpDes.append(1)
			else:
				tmpDes.append(0)

			if not planOnly:
				if rai_net.lgp.returnFeasible(BT.seqPath):
					tmpDes.append(1)
				else:
					tmpDes.append(0)
			else:
				tmpDes.append(-1)

			
			if not(rai_net.feas_net.modelFeasible is None and rai_net.feas2_net.modelFeasible is None):
				feasAct, feasSke, feasAct0, feasAct1, feasSke1, feasAct01 = rai_net.predictFeas(envState, rai_net.encodeAction(outDecoded))
				tmpDes.append(feasAct)
				tmpDes.append(feasSke)
				tmpDes.append(feasAct0)
				tmpDes.append(feasAct1)
				tmpDes.append(feasSke1)
				tmpDes.append(feasAct01)

			
				print("\t"+str(tmpDes[idxtmpdes])+" "+str(feasAct)+" "+str(feasSke)+" "+str(feasAct0)+" "+str(feasAct1)+" "+str(feasSke1)+" "+str(feasAct01))

			

		
		typeDecision.append(tmpDes)

		if planOnly:
			_, config = rai_world.applyKomo(komo, rai_net.logicalNames, num=komo.getPathFrames(rai_net.logicalNames).shape[0]-2, verbose=0)
		else:
			_, config = rai_world.applyKomo(komo, rai_net.logicalNames, num=komo.getPathFrames(rai_net.logicalNames).shape[0]-1, verbose=0)
		rai_net.K.setFrameState(config,verb=0)
		time.sleep(waitTime)
		
		
		if rai_net.lgp.isTerminal() or len(rai_net.preprocessGoals())==0:
			break
		""" else:
			print(rai_net.lgp.nodeState())
			print("\n")
			input("stop") """

		if depth==rai_net.maxDepth:
			print("Maximum depth reached: "+str(rai_net.maxDepth))
			successmsg="Maximum depth of "+str(rai_net.maxDepth)+" reached for goal"
			rai_net.K.setFrameState(X0, verb=0)
			break

		decisions=None

		if cheat_terminal:
			found, outDecoded = findTerminal(rai_net.lgp)
		else:
			found = False

		envState=rai_net.encodeState()
		if cheat_goalstate or rai_net.numGoal_orig>2:
			goals_tmp, unfullfilled, change=rai_net.preprocessGoalState(cheatGoalState=cheat_goalstate)
			inputState=rai_net.encodeInput(envState, goalState=goals_tmp)
			if change and rai_net.setup=="minimal":
				print("\tGoal changed to "+rai_net.goalString)
		else:
			
			inputState=rai_net.encodeInput(envState)



		if not found:
			tmpDes=[]
			outEncoded = rai_net.processPrediction(inputState)
			outDecoded = rai_net.decodeAction1(outEncoded)
			if cheat_feas and (rai_net.dataMode in [2,3]):
				old=outDecoded
				if (outDecoded not in rai_net.lgp.getDecisions() and not rai_net.lgp.getDecisions() == []):
					calcProb=False
				else:
					calcProb=True

				outDecoded, prob = findFeasible(rai_net, envState, outDecoded, Decisions=decisions, feasThresh=feasThresh, calcProb=calcProb)
				if old==outDecoded:
					print("NN Decision", outDecoded)
					tmpDes.append("NN")
					idxtmpdes=1

				else:
					print("New Feas Decision: ", outDecoded, "\twith probability", prob)
					print("\tInstead of: ", old)
					#tmpDes=[]
					tmpDes.append("FD")
					tmpDes.append(old)
					tmpDes.append(prob)
					idxtmpdes=3

			elif skeleton + " " +outDecoded in infeasibleSkeletons or cheat_tree or (outDecoded not in rai_net.lgp.getDecisions() and not rai_net.lgp.getDecisions() == [])or rai_net.checkPlaceSame(outDecoded):
				old=outDecoded
				if not skeleton + " " +outDecoded in infeasibleSkeletons:
					outDecoded, prob = rai_net.evalPredictions(inputState, prevSke=outDecoded)
				else:
					outDecoded, prob = rai_net.evalPredictions(inputState, infeasible=skeleton + " " +outDecoded, prevSke=outDecoded)
				print("New Decision: ", outDecoded, "\twith probability", prob)
				print("\tInstead of: ".expandtabs(4), old)
				#tmpDes=[]
				tmpDes.append("MP")
				tmpDes.append(old)
				tmpDes.append(prob)
				idxtmpdes=3
				#typeDecision.append(tmpDes)
			else:
				print("NN Decision", outDecoded)
				#tmpDes=[]
				tmpDes.append("NN")
				idxtmpdes=1
				#typeDecision.append(tmpDes)

		skeleton = skeleton + " " + outDecoded
		depth+=1

	if rai_net.dataMode in [11,12,13,14]:
		if not planOnly:
			feastmp = rai_net.lgp.returnFeasible(BT.seqPath)
		else:
			feastmp = False
	else:
		feastmp = rai_net.lgp.returnFeasible(BT.path)

	if len(rai_net.preprocessGoals())>0:
		print("--!! Goal not reached !!--")
		feastmp=False
	else:
		print("\nSkeleton found. Show path and node info")
		successmsg="Successfully reached goal"
		if planOnly:
			try:
				_ = rai_world.runLGP(rai_net.lgp, BT.seqPath, verbose=0, view=showFinal)
				feastmp = rai_net.lgp.returnFeasible(BT.seqPath)
			except:
				print("Can not solve komo for skeleton:", skeleton)
				rai_net.K.setFrameState(X0, verb=0)
				successmsg="KOMO failed for goal"

		elif showFinal and not rai_net.dataMode in [11,12,13,14]:
			rai_net.lgp.walkToRoot()
			rai_net.lgp.walkToNode(skeleton,0)
			#rai_net.K.setFrameState(X0)
			try:
				_ = rai_world.runLGP(rai_net.lgp, BT.path, verbose=0, view=showFinal)
			except:
				print("Can not solve komo for skeleton:", skeleton)
				rai_net.K.setFrameState(X0, verb=0)
				successmsg="KOMO failed for goal"
	if showFinal:
		rai_net.lgp.nodeInfo()
		
	rai_net.lgp.walkToRoot()
	rai_net.K.setFrameState(X0, verb=0)

	if not feastmp and rai_net.dataMode in [11,12,13,14]:
		print("--!! Infeasible Skeleton !!--")

	return skeleton, typeDecision, successmsg, feastmp

def main():
	#dir_file=os.path.abspath(os.path.dirname(__file__))


	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--rai_dir', type=str, default=dir_file)
	parser.add_argument('--verbose', type=int, default=1)
	parser.add_argument('--waitTime', type=float, default=0.01)

	parser.add_argument('--epochs_inst', type=int, default=500)
	parser.add_argument('--hlayers_inst', type=int, default=2)
	parser.add_argument('--hlayers_inst2', type=int, default=0)
	parser.add_argument('--size_inst', type=int, default=120)
	parser.add_argument('--epochs_grasp', type=int, default=1000)
	parser.add_argument('--hlayers_grasp', type=int, default=3)
	parser.add_argument('--size_grasp', type=int, default=64)
	parser.add_argument('--epochs_place', type=int, default=1000)
	parser.add_argument('--hlayers_place', type=int, default=3)
	parser.add_argument('--size_place', type=int, default=64)
	parser.add_argument('--num_batch_it', type=int, default=1)

	parser.add_argument('--epochs_feas', type=int, default=500)
	parser.add_argument('--hlayers_feas', type=int, default=2)
	parser.add_argument('--size_feas', type=int, default=64)

	parser.add_argument('--epochs_feas2', type=int, default=500)
	parser.add_argument('--hlayers_feas2', type=int, default=1)
	parser.add_argument('--hlayers_feas02', type=int, default=0)
	parser.add_argument('--size_feas2', type=int, default=64)

	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--lr_drop', type=float, default=1.0)
	parser.add_argument('--epoch_drop', type=int, default=100)
	parser.add_argument('--clipnorm', type=float, default=1.)
	parser.add_argument('--val_split', type=float, default=0.0)
	parser.add_argument('--reg_l2', type=float, default=0.0)
	parser.add_argument('--reg0_l2', type=float, default=0.0)


	parser.add_argument('--train_only', dest='train_only', action='store_true')
	parser.set_defaults(train_only=False)
	parser.add_argument('--feas_only', dest='feas_only', action='store_true')
	parser.set_defaults(feas_only=False)
	parser.add_argument('--feas2_only', dest='feas2_only', action='store_true')
	parser.set_defaults(feas2_only=False)
	parser.add_argument('--saveModel', dest='saveModel', action='store_true')
	parser.set_defaults(saveModel=False)
	parser.add_argument('--model_dir', type=str, default='')
	parser.add_argument('--model_dir_data', type=str, default='')

	parser.add_argument('--cheat_terminal', dest='cheat_terminal', action='store_true')
	parser.set_defaults(cheat_terminal=False)
	parser.add_argument('--cheat_tree', dest='cheat_tree', action='store_true')
	parser.set_defaults(cheat_tree=False)
	parser.add_argument('--cheat_feas', dest='cheat_feas', action='store_true')
	parser.set_defaults(cheat_feas=False)
	parser.add_argument('--cheat_goalstate', dest='cheat_goalstate', action='store_true')
	parser.set_defaults(cheat_goalstate=False)
	parser.add_argument('--completeTesting', dest='completeTesting', action='store_true')
	parser.set_defaults(completeTesting=False)
	parser.add_argument('--allEnv', dest='allEnv', action='store_true')
	parser.set_defaults(allEnv=False)
	parser.add_argument('--showFinal', dest='showFinal', action='store_true')
	parser.set_defaults(showFinal=False)
	parser.add_argument('--viewConfig', dest='viewConfig', action='store_true')
	parser.set_defaults(viewConfig=False)
	parser.add_argument('--planOnly', dest='planOnly', action='store_true')
	parser.set_defaults(planOnly=False)


	parser.add_argument('--feasThresh', type=float, default=0.4)
	parser.add_argument('--feasWeight', type=float, default=0.5)


	parser.add_argument('--goal', type=str, default="(on red green) (on green blue)")
	parser.add_argument('--env', type=int, default=1)
	parser.add_argument('--setup', type=str, default="minimal")
	parser.add_argument('--NNmode', type=str, default="minimal")
	parser.add_argument('--datasetMode', type=int, default=1)
	parser.add_argument('--start', type=int, default=1)
	parser.add_argument('--startSub', type=int, default=1)
	parser.add_argument('--maxDepth', type=int, default=20)
	
	args = parser.parse_args()
	path_rai = args.rai_dir
	verbose=args.verbose
	waitTime= args.waitTime

	train_only = args.train_only
	feas_only = args.feas_only
	feas2_only = args.feas2_only


	saveModel = args.saveModel
	cheat_terminal = args.cheat_terminal
	cheat_goalstate= args.cheat_goalstate
	cheat_tree=args.cheat_tree
	cheat_feas=args.cheat_feas

	feasThresh=args.feasThresh
	feasWeight=args.feasWeight


	completeTesting=args.completeTesting
	allEnv=args.allEnv
	showFinal=args.showFinal
	viewConfig=args.viewConfig
	planOnly=args.planOnly

	model_dir=args.model_dir
	model_dir_data=args.model_dir_data
	if model_dir_data=="":
		model_dir_data=model_dir
	
	epochs_inst=args.epochs_inst
	n_size_inst=args.size_inst
	n_layers_inst=args.hlayers_inst
	n_layers_inst2=args.hlayers_inst2

	epochs_grasp=args.epochs_grasp
	n_size_grasp=args.size_grasp
	n_layers_grasp=args.hlayers_grasp
	epochs_place=args.epochs_place
	n_size_place=args.size_place
	n_layers_place=args.hlayers_place
	num_batch_it=args.num_batch_it

	epochs_feas=args.epochs_feas
	n_size_feas=args.size_feas
	n_layers_feas=args.hlayers_feas

	epochs_feas2=args.epochs_feas2
	n_size_feas2=args.size_feas2
	n_layers_feas2=args.hlayers_feas2
	n_layers_feas02=args.hlayers_feas02


	lr=args.lr
	lr_drop=args.lr_drop
	epoch_drop=args.epoch_drop
	clipnorm=args.clipnorm
	val_split=args.val_split
	reg=args.reg_l2
	reg0=args.reg0_l2

	nenv=args.env
	goalString=args.goal
	setup=args.setup
	NNmode=args.NNmode
	dataMode=args.datasetMode
	maxDepth=args.maxDepth
	start=args.start
	startSub=args.startSub
	start0=start

	if not dataMode in [11,12,13,14]:
		planOnly=False

	#-------------------------------------------------------------------------------------------------------------------------	
	print("Setting up basic Config and FOL for env: "+str(nenv))
	rai=rai_world.RaiWorld(path_rai, nenv, setup, goalString, verbose, maxDepth=maxDepth, NNmode=NNmode, datasetMode=dataMode, view=viewConfig)

	print("\nModel and dataset")
	if saveModel and not (feas_only or feas2_only):
		rai.saveFit(model_dir_data,epochs_inst, n_layers_inst, n_size_inst, epochs_grasp, n_layers_grasp, n_size_grasp, epochs_place, n_layers_place, n_size_place,
					lr, lr_drop, epoch_drop, clipnorm, val_split, reg, reg0, num_batch_it, n_layers_inst2=n_layers_inst2)
		print("Model trained: "+rai.rai_net.timestamp)
	else:
		rai.loadFit(model_dir)
		print("Model loaded")


	if dataMode in [3,2] or NNmode in ["mixed3", "mixed2"] or (dataMode in [11,12,13,14] and (feas_only or planOnly)):
		if saveModel:
			if not feas2_only:
				rai.saveFeas(model_dir_data,epochs_feas, n_layers_feas, n_size_feas, lr, lr_drop, epoch_drop,clipnorm, val_split, reg0)
				print("Feasible model 1 trained: "+rai.model_dir)
			rai.saveFeas2(model_dir_data,epochs_feas2, n_layers_feas2, n_size_feas2, lr, lr_drop, epoch_drop,clipnorm, val_split, reg, reg0, num_batch_it, n_layers_feas2=n_layers_feas02)
			print("Feasible model 2 trained: "+rai.model_dir)
		else:
			rai.loadFeas()
			rai.loadFeas2()
			print("Feasible model loaded")


	if not train_only and not feas_only and not feas2_only and not model_dir=="":
		if planOnly:
			append_test="_plan"
		else:
			append_test=""
		if cheat_feas:
			rai.feasweight=feasWeight
		if completeTesting:
			if allEnv:
				rangeEnv=range(nenv,104)
			else:
				rangeEnv=range(nenv,nenv+1)

			for nenv in rangeEnv:
				summary=[[],[],[],[],[]] #optimal feasible infeasible-op infeasible no

				start=start0
				rai.nenv=nenv
				path=createResults(rai,cheat_tree=cheat_tree, cheat_terminal = cheat_terminal, cheat_goalstate=cheat_goalstate, cheat_feas=cheat_feas, start=start0, feasThresh=feasThresh, feasWeight=feasWeight, planOnly=planOnly)
				if 2*(start-1) < len(minimal_experiment.Sets):
					numGoal=start-1
					if not startSub==1:
						numGoal+=1
					for i in range(2*(start-1)+startSub-1,len(minimal_experiment.Sets)):
						if i%2==0:
							numGoal+=1
							strgoal=str(numGoal).zfill(3)+"-1"
						else:
							strgoal=str(numGoal).zfill(3)+"-2"

						goal=minimal_experiment.Sets[i]

						infeasibleSkeletons=[]
						for tries in range(4):
							rai.redefine(goal, nenv=nenv)
							rai.resetFit()
							print("----Test Goal "+strgoal+": '"+rai.goalString_orig+"' for env "+str(rai.nenv)+"----\n")
							skeleton, typeDecision,successmsg, feasible=buildSkeleton(rai, cheat_tree=cheat_tree, cheat_terminal = cheat_terminal, cheat_goalstate=cheat_goalstate, showFinal=showFinal, waitTime=waitTime, cheat_feas=cheat_feas, feasThresh=feasThresh, planOnly=planOnly, infeasibleSkeletons=infeasibleSkeletons)
							infeasibleSkeletons= infeasibleSkeletons + rai_world.splitStringPath(skeleton, list_old=[])
							writeResults(rai,skeleton,typeDecision,successmsg,path,goalnumber_string=strgoal, planOnly=planOnly, feasible=feasible, tries=tries)

							if feasible:
								break
						printResult(rai, skeleton)

						if i%2==0:
							idxsol=4
							skeletonPrev=skeleton
							strgoalPrev=strgoal
							if successmsg=="Successfully reached goal":
								solutions, _, _ = minimal_experiment.getData(nenv=nenv, nset=numGoal)
								if not solutions ==[]:
									if not feasible:
										if len(rai_world.splitStringStep(solutions[0], list_old=[])) == len(rai_world.splitStringStep(skeleton, list_old=[])):
											idxsol=2
										else:
											idxsol=3
									elif len(rai_world.splitStringStep(solutions[0], list_old=[])) == len(rai_world.splitStringStep(skeleton, list_old=[])):
										idxsol=0
									else:
										idxsol=1
						else:
							if skeleton == skeletonPrev:
								feasible = feasible or idxsol in [0,1]
								#input(feasible)
								if feasible and idxsol in [2,3]:
									idxsol=idxsol-2
							else:
								summary[idxsol].append(strgoalPrev)

							if successmsg=="Successfully reached goal":
								solutions, _, _ = minimal_experiment.getData(nenv=nenv, nset=numGoal)
								if not solutions ==[]:
									if not feasible:
										if len(rai_world.splitStringStep(solutions[0], list_old=[])) == len(rai_world.splitStringStep(skeleton, list_old=[])):
											summary[2].append(strgoal)
										else:
											summary[3].append(strgoal)
									elif len(rai_world.splitStringStep(solutions[0], list_old=[])) == len(rai_world.splitStringStep(skeleton, list_old=[])):
										summary[0].append(strgoal)
									else:
										summary[1].append(strgoal)
									
									if skeleton == skeletonPrev:
										summary[idxsol].append(strgoalPrev)
							else:
								summary[4].append(strgoal)

					start=0
				else:
					numGoal=start-1
					start=(start-1)-len(minimal_experiment.Sets)/2
					#print(start)

				for i in range(int(start),len(minimal_experiment.test)):
					numGoal+=1
					strgoal=str(numGoal).zfill(3)+"-1"

					goal= minimal_experiment.test[i]+" "+minimal_experiment.test[i]
					infeasibleSkeletons=[]
					for tries in range(4):
						rai.redefine(goal, nenv=nenv)
						rai.resetFit()
						print("----Test Goal "+strgoal+": '"+rai.goalString_orig+"' for env "+str(rai.nenv)+"----\n")

						skeleton, typeDecision,successmsg, feasible=buildSkeleton(rai,cheat_tree=cheat_tree, cheat_terminal = cheat_terminal, cheat_goalstate=cheat_goalstate, showFinal=showFinal,waitTime=waitTime, cheat_feas=cheat_feas, feasThresh=feasThresh, planOnly=planOnly)
						infeasibleSkeletons= infeasibleSkeletons + rai_world.splitStringPath(skeleton, list_old=[])
						writeResults(rai,skeleton,typeDecision,successmsg,path, goalnumber_string=strgoal, planOnly=planOnly, feasible=feasible)
						if feasible:
							break
					printResult(rai, skeleton)

					if successmsg=="Successfully reached goal":
						solutions, _, _ = minimal_experiment.getData1(nenv=nenv, nset=i+1)
						if not solutions ==[]:
							if not feasible:
								if len(rai_world.splitStringStep(solutions[0], list_old=[])) == len(rai_world.splitStringStep(skeleton, list_old=[])):
									summary[2].append(strgoal)
								else:
									summary[3].append(strgoal)
							elif len(rai_world.splitStringStep(solutions[0], list_old=[])) == len(rai_world.splitStringStep(skeleton, list_old=[])):
								summary[0].append(strgoal)
							else:
								summary[1].append(strgoal)

					else:
						summary[4].append(strgoal)
				
				with open(path+'/Summaryenv'+str(nenv).zfill(3)+append_test+'.txt', 'a+') as f:
					f.write(rai.NNmode+", datamode "+str(rai.dataMode)+", feas threshold "+str(feasThresh)+", feas weight "+str(feasWeight)+", maxDepth "+str(maxDepth)+"\n\n" )
					f.write("----Optimal Solution ("+str(len(summary[0]))+")----\n")
					for elem in summary[0]:
						f.write(elem+"\t")
					f.write("\n----Feasible Solution ("+str(len(summary[1]))+")----\n")
					for elem in summary[1]:
						f.write(elem+"\t")
					f.write("\n----Infeasible-op Solution ("+str(len(summary[2]))+")----\n")
					for elem in summary[2]:
						f.write(elem+"\t")
					f.write("\n----Infeasible Solution ("+str(len(summary[3]))+")----\n")
					for elem in summary[3]:
						f.write(elem+"\t")
					f.write("\n----No Solution ("+str(len(summary[4]))+")----\n")
					for elem in summary[4]:
						f.write(elem+"\t")
					f.write("\n")

		else:
			print("\nPredict: "+goalString)
			path=createResults(rai,cheat_tree=cheat_tree, cheat_terminal = cheat_terminal, cheat_goalstate=cheat_goalstate, cheat_feas=cheat_feas, start=start0, feasThresh=feasThresh, feasWeight=feasWeight, planOnly=planOnly, test=True)
			skeleton,typeDecision,successmsg, feasible=buildSkeleton(rai, cheat_terminal = cheat_terminal, cheat_goalstate=cheat_goalstate, cheat_feas=cheat_feas, feasThresh=feasThresh, planOnly=planOnly)
			printResult(rai, skeleton)
			writeResults(rai,skeleton,typeDecision,successmsg,path,goalnumber_string="", planOnly=planOnly, feasible=feasible, tries=0, test=True)
			if not feasible:
				print("Infeasible Solution")
			if saveModel and successmsg=="Successfully reached goal":
				with open(path_rai+'/logs/toTest.txt', 'a+') as f:
					f.write(rai.model_dir+'\n')

	if not completeTesting and not (train_only or feas_only):
		input("Press Enter to end Program...")



if __name__ == "__main__":
    main()
