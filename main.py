#!/usr/bin/python3

# python test_1.py --rai_dir ~/rai-python
import sys
import os

dir_file=os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir_file+'/../ry/')
from libry import *


#from numpy import *

import tensorflow as tf

import rai_setup as rai_world
import minimal_experiment

import datetime
import time
import shutil


def createResults(rai_net, cheat_goalstate=False,cheat_tree=False, start=1, planOnly=False, test=False):
	if rai_net.setup == "minimal":
		append_folder=""
		if cheat_goalstate:
			append_folder=append_folder+"_goalstate"
		if cheat_tree:
			append_folder=append_folder+"_tree"

		
		path=rai_net.path_rai+'/result_minimal/'+rai_net.model_dir+append_folder
		if not(os.path.exists(path)):
				os.makedirs(path)
				
		if start==1:
			if rai_net.NNmode in ["mixed10"]:
				appendmode="mixed"+str(rai_net.dataMode)
			elif rai_net.NNmode in ["FFnew", "minimal"]:
				appendmode=rai_net.NNmode+str(rai_net.dataMode)
			else:
				NotImplementedError

			if planOnly:
				append_test="_plan"
			else:
				append_test=""
			if test:
				append_test="_test"

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
					f.write("[ "+str(typedec[1])+" "+str(typedec[2])+" ]")
					f.write("\n")

				elif typedec[0]=="MP":
					f.write("[ "+str(typedec[3])+" "+str(typedec[4])+" ]")	
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
					infeasibleSkeletons=[], depthSkeletons=[], tries=0, verbose=True):

	skeleton=""
	typeDecision=[]
	tmpDes=[]
	depth=0
	#X0 = rai_net.K.getFrameState()
	K0=Config()
	K0.copy(rai_net.K)

	if tries>0:
		print("Next try: "+str(tries))
		#print(infeasibleSkeletons)

	envState=rai_net.encodeState()
	if cheat_goalstate or rai_net.numGoal_orig>2:
			goals_tmp, unfullfilled, change=rai_net.preprocessGoalState(cheatGoalState=cheat_goalstate)
			inputState=rai_net.encodeInput(envState, goalState=goals_tmp)
			if change and rai_net.setup=="minimal" and verbose:
				print("\tGoal changed to "+unfullfilled[0])
	else:
		
		inputState=rai_net.encodeInput(envState)

	#lgp.viewTree()

	successmsg="Failed to reach goal"

	if rai_net.lgp.isTerminal() or len(rai_net.preprocessGoals())==0:
		return skeleton, typeDecision, "Successfully reached goal", True

	while True:

		tmpDes=[]
		outEncoded = rai_net.processPrediction(inputState)
		outDecoded = rai_net.decodeAction1(outEncoded)
		if skeleton + " " +outDecoded in infeasibleSkeletons+depthSkeletons or (outDecoded in infeasibleSkeletons+depthSkeletons and depth==0) or cheat_tree or (outDecoded not in rai_net.lgp.getDecisions() and not rai_net.lgp.getDecisions() == [])or rai_net.checkPlaceSame(outDecoded):
			old=outDecoded
			infeasibleSke=[]
			depthSke=[]
			if depth==0:
				for tmpske in infeasibleSkeletons:
					tmpskeStep = rai_world.splitStringStep(tmpske, list_old=[])
					if len(tmpskeStep)==1:
						infeasibleSke.append(tmpskeStep[-1])
				for tmpske in depthSkeletons:
					tmpskeStep = rai_world.splitStringStep(tmpske, list_old=[])
					if len(tmpskeStep)==1:
						depthSke.append(tmpskeStep[-1])

			else:
				for tmpske in infeasibleSkeletons:
					tmpskeStep = rai_world.splitStringStep(tmpske, list_old=[])
					if skeleton == ' '.join(tmpskeStep[:-1]):
						infeasibleSke.append(tmpskeStep[-1])
				for tmpske in depthSkeletons:
					tmpskeStep = rai_world.splitStringStep(tmpske, list_old=[])
					if skeleton == ' '.join(tmpskeStep[:-1]):
						depthSke.append(tmpskeStep[-1])


			outDecoded, prob = rai_net.evalPredictions(inputState, infeasible=infeasibleSke, maxdepth=depthSke, prevSke=outDecoded, depth=depth+1, tries=tries)
			if verbose:
				print("MP Decision", outDecoded, "\twith probability", prob)
				print("\tInstead of: ".expandtabs(4), old)
			#tmpDes=[]
			tmpDes.append("MP")
			tmpDes.append(old)
			tmpDes.append(prob)
			#typeDecision.append(tmpDes)
		else:
			if verbose:
				print("NN Decision", outDecoded)
			#tmpDes=[]
			tmpDes.append("NN")
			#typeDecision.append(tmpDes)

		if skeleton=="":
			skeleton=outDecoded
		else:
			skeleton = skeleton + " " + outDecoded
		depth+=1


		rai_net.lgp.walkToRoot()
		rai_net.lgp.walkToNode(skeleton,0)


		try:
			komo = rai_world.runLGP(rai_net.lgp, BT.seq, verbose=0, view=False)
			if not planOnly:
				komo = rai_world.runLGP(rai_net.lgp, BT.seqPath, verbose=0, view=False)

		except:
			print("Can not solve komo path for skeleton:", skeleton)
			successmsg="KOMO path failed for goal"
			rai_net.lgp.walkToRoot()
			#rai_net.K.setFrameState(X0, verb=0)
			rai_net.K.copy(K0)
			break

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

		typeDecision.append(tmpDes)

		#K2=Config()
		if planOnly:
			#_, config = rai_world.applyKomo(komo, rai_net.logicalNames, num=komo.getPathFrames(rai_net.logicalNames).shape[0]-2, verbose=0)
			komo.getKFromKomo(rai_net.K, komo.getPathFrames(rai_net.logicalNames).shape[0]-2)
		else:
			#_, config = rai_world.applyKomo(komo, rai_net.logicalNames, num=komo.getPathFrames(rai_net.logicalNames).shape[0]-1, verbose=0)
			komo.getKFromKomo(rai_net.K, komo.getPathFrames(rai_net.logicalNames).shape[0]-1)
		#rai_net.K.setFrameState(config,verb=0)

		#rai_net.K.copy(K2)

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
			#rai_net.K.setFrameState(X0, verb=0)
			rai_net.K.copy(K0)
			break


		envState=rai_net.encodeState()
		if cheat_goalstate or rai_net.numGoal_orig>2:
			goals_tmp, unfullfilled, change=rai_net.preprocessGoalState(cheatGoalState=cheat_goalstate)
			inputState=rai_net.encodeInput(envState, goalState=goals_tmp)
			if change and rai_net.setup=="minimal" and verbose:
				print("\tGoal changed to "+rai_net.goalString)
		else:
			
			inputState=rai_net.encodeInput(envState)

	if not planOnly:
		feastmp = rai_net.lgp.returnFeasible(BT.seqPath)
	else:
		feastmp = False


	if len(rai_net.preprocessGoals())>0:
		if verbose:
			print("--!! Goal not reached !!--")
		feastmp=False
	else:
		if verbose:
			print("\nSkeleton found. Show path and node info")
		successmsg="Successfully reached goal"
		if planOnly:
			try:
				_ = rai_world.runLGP(rai_net.lgp, BT.seqPath, verbose=0, view=showFinal)
				feastmp = rai_net.lgp.returnFeasible(BT.seqPath)
			except:
				print("Can not solve komo for skeleton:", skeleton)
				#rai_net.K.setFrameState(X0, verb=0)
				rai_net.K.copy(K0)
				successmsg="KOMO failed for goal"

	if showFinal:
		rai_net.lgp.nodeInfo()
		
	rai_net.lgp.walkToRoot()
	#rai_net.K.setFrameState(X0, verb=0)
	rai_net.K.copy(K0)

	if not feastmp and verbose:
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
	parser.add_argument('--batch_size', type=int, default=32)

	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--lr_drop', type=float, default=1.0)
	parser.add_argument('--epoch_drop', type=int, default=100)
	parser.add_argument('--clipnorm', type=float, default=100.0)
	parser.add_argument('--val_split', type=float, default=0.0)
	parser.add_argument('--reg_l2', type=float, default=0.0)
	parser.add_argument('--reg0_l2', type=float, default=0.0)


	parser.add_argument('--train_only', dest='train_only', action='store_true')
	parser.set_defaults(train_only=False)
	parser.add_argument('--saveModel', dest='saveModel', action='store_true')
	parser.set_defaults(saveModel=False)
	parser.add_argument('--model_dir', type=str, default='')
	parser.add_argument('--model_dir_data', type=str, default='')

	parser.add_argument('--cheat_tree', dest='cheat_tree', action='store_true')
	parser.set_defaults(cheat_tree=False)
	parser.add_argument('--cheat_goalstate', dest='cheat_goalstate', action='store_true')
	parser.set_defaults(cheat_goalstate=False)
	parser.add_argument('--completeTesting', dest='completeTesting', action='store_true')
	parser.set_defaults(completeTesting=False)
	parser.add_argument('--completeTraining', dest='completeTraining', action='store_true')
	parser.set_defaults(completeTraining=False)
	parser.add_argument('--allEnv', dest='allEnv', action='store_true')
	parser.set_defaults(allEnv=False)
	parser.add_argument('--showFinal', dest='showFinal', action='store_true')
	parser.set_defaults(showFinal=False)
	parser.add_argument('--viewConfig', dest='viewConfig', action='store_true')
	parser.set_defaults(viewConfig=False)
	parser.add_argument('--planOnly', dest='planOnly', action='store_true')
	parser.set_defaults(planOnly=False)
	parser.add_argument('--exclude', dest='exclude', action='store_true')
	parser.set_defaults(exclude=False)

	parser.add_argument('--goal', type=str, default="(on red green) (on green blue)")
	parser.add_argument('--env', type=int, default=1)
	parser.add_argument('--setup', type=str, default="minimal")
	parser.add_argument('--NNmode', type=str, default="minimal")
	parser.add_argument('--datasetMode', type=int, default=1)
	parser.add_argument('--start', type=int, default=1)
	parser.add_argument('--startSub', type=int, default=1)
	parser.add_argument('--maxDepth', type=int, default=20)
	parser.add_argument('--maxTries', type=int, default=4)
	
	args = parser.parse_args()
	path_rai = args.rai_dir
	verbose=args.verbose
	waitTime= args.waitTime

	train_only = args.train_only

	saveModel = args.saveModel
	cheat_goalstate= args.cheat_goalstate
	cheat_tree=args.cheat_tree

	completeTesting=args.completeTesting
	completeTraining=args.completeTraining
	allEnv=args.allEnv
	showFinal=args.showFinal
	viewConfig=args.viewConfig
	planOnly=args.planOnly
	exclude=args.exclude

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
	batch_size=args.batch_size

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
	maxTries=args.maxTries
	start=args.start
	startSub=args.startSub
	start0=start

	if not dataMode in [1,2,3,4]:
		planOnly=False

	obg = [[],[]]#[[15], [10,20,30,39,48,53,58,63,68]]
	or1 = [[2], [11,12,13,14,15,16,17,18,19,20]]
	og2 = [[8], [3,13,23,33,42,59,60,61,62,63]]

	#-------------------------------------------------------------------------------------------------------------------------	
	print("Setting up basic Config and FOL for env: "+str(nenv))
	rai=rai_world.RaiWorld(path_rai, nenv, setup, goalString, verbose, maxDepth=maxDepth, NNmode=NNmode, datasetMode=dataMode, view=viewConfig)

	print("\nModel and dataset")
	if saveModel:
		rai.saveFit(model_dir_data,epochs_inst, n_layers_inst, n_size_inst, epochs_grasp, n_layers_grasp, n_size_grasp, epochs_place, n_layers_place, n_size_place,
					lr, lr_drop, epoch_drop, clipnorm, val_split, reg, reg0, num_batch_it, batch_size,n_layers_inst2=n_layers_inst2)
		print("Model trained: "+rai.rai_net.timestamp)
	else:
		rai.loadFit(model_dir)
		print("Model loaded")

	if not train_only and not model_dir=="":
		if planOnly:
			append_test="_plan"
		else:
			append_test=""

		if completeTesting or completeTraining:
			if completeTraining:
				if allEnv:
					rangeEnv=[29,38,46,56,64,73,82,91,
								32,39,48,57,65,75,83,93]#,
								#33,40,50,58,66,78,88,94,
								#35,42,51,59,68,80,89,97,
								#43,54,63,72,81,90]
					rangeEnv=rangeEnv[rangeEnv.index(nenv):]
				else:
					rangeEnv=range(nenv,nenv+1)
			else:
				if allEnv:
					if nenv in range(100,104):
						rangeEnv=range(nenv,104)
					else:
						rangeEnv=range(nenv,110)
				else:
					rangeEnv=range(nenv,nenv+1)


			for nenv in rangeEnv:

				if nenv ==109:
					rai.maxDepth=rai.maxDepth+2
					maxTries=maxTries+2

				summary=[[],[],[],[],[]] #optimal feasible infeasible-op infeasible no

				start=start0
				rai.nenv=nenv
				path=createResults(rai,cheat_tree=cheat_tree, cheat_goalstate=cheat_goalstate, start=start0, planOnly=planOnly)
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

						if exclude and not numGoal in or1[1]+og2[1]+obg[1]:
							#print("skip "+str(numGoal))
							continue

						goal=minimal_experiment.Sets[i]

						infeasibleSkeletons=[]
						depthSkeletons=[]
						rai.redefine(goal, nenv=nenv)

						for tries in range(maxTries):
							rai.resetFit(cheatGoalState=cheat_goalstate, goal=goal)
							print("----Test Goal "+strgoal+": '"+rai.goalString_orig+"' for env "+str(rai.nenv)+"----\n")
							skeleton, typeDecision,successmsg, feasible=buildSkeleton(rai, cheat_tree=cheat_tree, cheat_goalstate=cheat_goalstate, showFinal=showFinal, waitTime=waitTime, planOnly=planOnly, infeasibleSkeletons=infeasibleSkeletons, depthSkeletons=depthSkeletons, tries=tries)
							if successmsg=="Maximum depth of "+str(rai.maxDepth)+" reached for goal":
								depthSkeletons= depthSkeletons + rai_world.splitStringPath(skeleton, list_old=[])
							else:
								infeasibleSkeletons= infeasibleSkeletons + rai_world.splitStringPath(skeleton, list_old=[])
							writeResults(rai,skeleton,typeDecision,successmsg,path,goalnumber_string=strgoal, planOnly=planOnly, feasible=feasible, tries=tries)

							if feasible or skeleton=="":
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
										if len(rai_world.splitStringStep(solutions[0], list_old=[])) >= len(rai_world.splitStringStep(skeleton, list_old=[])):
											idxsol=2
										else:
											idxsol=3
									elif len(rai_world.splitStringStep(solutions[0], list_old=[])) >= len(rai_world.splitStringStep(skeleton, list_old=[])):
										idxsol=0
									else:
										idxsol=1
							else:
								summary[4].append(strgoal)
								idxsol=4
						else:
							if skeleton == skeletonPrev:
								feasible = feasible or idxsol in [0,1]
								#input(feasible)
								if feasible and idxsol in [2,3]:
									idxsol=idxsol-2
							elif not idxsol==4:
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

				for i in range(int(start),len(minimal_experiment.test)):
					numGoal+=1
					if exclude and not i+1 in or1[0]+og2[0]+obg[0]:
						#print("skip "+str(numGoal))
						continue

					strgoal=str(numGoal).zfill(3)+"-1"

					goal= minimal_experiment.test[i]+" "+minimal_experiment.test[i]
					infeasibleSkeletons=[]
					depthSkeletons=[]
					rai.redefine(goal, nenv=nenv)

					for tries in range(maxTries):
						rai.resetFit(cheatGoalState=cheat_goalstate, goal=goal)
						print("----Test Goal "+strgoal+": '"+rai.goalString_orig+"' for env "+str(rai.nenv)+"----\n")

						skeleton, typeDecision,successmsg, feasible=buildSkeleton(rai,cheat_tree=cheat_tree, cheat_goalstate=cheat_goalstate, showFinal=showFinal,waitTime=waitTime, planOnly=planOnly, infeasibleSkeletons=infeasibleSkeletons, depthSkeletons=depthSkeletons ,tries=tries)
						if successmsg=="Maximum depth of "+str(rai.maxDepth)+" reached for goal":
							depthSkeletons= depthSkeletons + rai_world.splitStringPath(skeleton, list_old=[])
						else:
							infeasibleSkeletons= infeasibleSkeletons + rai_world.splitStringPath(skeleton, list_old=[])
						writeResults(rai,skeleton,typeDecision,successmsg,path, goalnumber_string=strgoal, planOnly=planOnly, feasible=feasible)
						if feasible:
							break
					printResult(rai, skeleton)

					if successmsg=="Successfully reached goal":
						solutions, _, _ = minimal_experiment.getData1(nenv=nenv, nset=i+1)
						if not solutions ==[]:
							if not feasible:
								if len(rai_world.splitStringStep(solutions[0], list_old=[])) >= len(rai_world.splitStringStep(skeleton, list_old=[])):
									summary[2].append(strgoal)
								else:
									summary[3].append(strgoal)
							elif len(rai_world.splitStringStep(solutions[0], list_old=[])) >= len(rai_world.splitStringStep(skeleton, list_old=[])):
								summary[0].append(strgoal)
							else:
								summary[1].append(strgoal)

					else:
						summary[4].append(strgoal)
				
				with open(path+'/Summaryenv'+str(nenv).zfill(3)+append_test+'.txt', 'a+') as f:
					f.write(rai.NNmode+", datamode "+str(rai.dataMode)+", maxDepth "+str(rai.maxDepth)+", maxTries "+str(maxTries)+"\n\n" )
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
			path=createResults(rai,cheat_tree=cheat_tree, cheat_goalstate=cheat_goalstate, start=start0, planOnly=planOnly, test=True)
			infeasibleSkeletons=[]
			depthSkeletons=[]
			teststep=1
			starttime=time.time()
			for tries in range(maxTries):
				rai.resetFit(cheatGoalState=cheat_goalstate, goal=goalString)
				skeleton,typeDecision,successmsg, feasible=buildSkeleton(rai, cheat_goalstate=cheat_goalstate, planOnly=planOnly, infeasibleSkeletons=infeasibleSkeletons, depthSkeletons=depthSkeletons, verbose=False, showFinal=showFinal)

				if successmsg=="Maximum depth of "+str(rai.maxDepth)+" reached for goal":
					depthSkeletons= depthSkeletons + rai_world.splitStringPath(skeleton, list_old=[])
				else:
					infeasibleSkeletons= infeasibleSkeletons + rai_world.splitStringPath(skeleton, list_old=[])

				writeResults(rai,skeleton,typeDecision,successmsg,path,goalnumber_string="", planOnly=planOnly, feasible=feasible, tries=0, test=True)
				if feasible:
					break
				teststep=teststep+1
			endtime=time.time()
			print("time: "+str(endtime-starttime)+", tries: "+str(teststep))

			printResult(rai, skeleton)
			if not feasible:
				print("Infeasible Solution")
			if saveModel and successmsg=="Successfully reached goal":
				with open(path_rai+'/logs/toTest.txt', 'a+') as f:
					f.write(rai.model_dir+'\n')

	if not completeTesting and not train_only and not completeTraining:
		input("Press Enter to end Program...")



if __name__ == "__main__":
    main()
