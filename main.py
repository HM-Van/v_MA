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

# Main function
# Train hierarchical policy
# Use hierarchical policy to solve sequential manipulation problems
# Note: goal formulations: (held obj) (on obj table)

def createResults(rai_net, cheat_goalstate=False,cheat_tree=False, start=1, planOnly=False, test=False):
	# Creates directory for results
	if rai_net.setup == "minimal":

		# Append modes to name
		append_folder=""
		if cheat_goalstate:
			append_folder=append_folder+"_goalstate"
		if cheat_tree:
			append_folder=append_folder+"_tree"

		# Create dir
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
			
			# Append mode to file name
			if planOnly:
				append_test="_plan"
			else:
				append_test=""
			if test:
				append_test=append_test+"_test"

			# Copy parameter file
			shutil.copyfile(rai_net.path_rai+'/logs/'+rai_net.model_dir+"_"+appendmode+'/params.txt',path+'/params.txt')

			# Create Summary file
			with open(path+'/env'+str(rai_net.nenv).zfill(3)+append_test+'.txt', 'a+') as f:
				f.write("\n\n-----Result for env "+str(rai_net.nenv)+"-----\n\n")

		return path

def writeResults(rai_net,skeleton,typeDecision,successmsg,path, goalnumber_string="", planOnly=False, feasible=True, tries=0, test=False):
	
	if rai_net.setup == "minimal":
		# Append mode to file name
		if planOnly:
			append_test="_plan"
		else:
			append_test=""
		if test:
			append_test=append_test+"_test"

		
		if feasible:
			strfeas=" feasible 1"
		else:
			strfeas=" infeasible 0"

		with open(path+'/env'+str(rai_net.nenv).zfill(3)+append_test+'.txt', 'a+') as f:
			# Header_ objective, feasible
			f.write("----"+successmsg+" "+goalnumber_string+": "+rai_net.goalString_orig+strfeas+"----\n")
			if tries>0 and feasible:
				f.write("---- in "+str(tries)+" tries ----\n")

			actions=rai_world.splitStringStep(skeleton, list_old=[],verbose=0)
			for act, typedec in zip(actions,typeDecision):
				# For all high-level actions
				# Print action
				f.write("\t\t  "+typedec[0]+": "+act)
				if typedec[0]=="NN":
					# If directly through policy [seq seqPath] feasibility
					f.write("[ "+str(typedec[1])+" "+str(typedec[2])+" ]")
					f.write("\n")

				elif typedec[0]=="MP":
					# If through evalution of all high-level actions: [seq seqPath] feasibility + originahigh-level action that was replaced
					f.write("[ "+str(typedec[3])+" "+str(typedec[4])+" ]")	
					f.write("\twith prob "+str(round(typedec[2],3))+" instead of "+typedec[1]+"\n")

				else:
					NotImplementedError
			f.write("--------------------\n\n")
			
		
	
def printResult(rai_net, skeleton):
	# Print final skeleton to console
	print("\n----Final Result----")
	print("Goal\t\t: "+rai_net.goalString_orig)
	print("Final skeleton\t: ")

	actions=rai_world.splitStringStep(skeleton, list_old=[],verbose=0)
	for act in actions:
		print("\t\t  "+act)
	print("--------------------")
	print("")

def buildSkeleton(rai_net, cheat_terminal = False, cheat_goalstate=False,cheat_tree=False, showFinal=True, waitTime=0, cheat_feas=False, feasThresh=0.4, planOnly=False,
					infeasibleSkeletons=[], depthSkeletons=[], tries=0, verbose=True, newLGP=True):

	skeleton=""
	tmpskeleton=""
	typeDecision=[]
	tmpDes=[]
	depth=0
	
	K0=Config()
	K0.copy(rai_net.K)

	timeHP=0
	timeOP=0

	if tries>0:
		print("Next try: "+str(tries))

	successmsg="Failed to reach goal"

	if rai_net.lgp.isTerminal() or len(rai_net.preprocessGoals())==0:
		return skeleton, typeDecision, "Successfully reached goal", True

	while True:
		starttime=time.time()
		tmpDes=[]
		# Get encoded state
		envState=rai_net.encodeState()
		if cheat_goalstate or rai_net.numGoal_orig>2:
			# Get (possibly new) encoded objective
			goals_tmp, _, change=rai_net.preprocessGoalState(cheatGoalState=cheat_goalstate)
			# Get encoded input
			inputState=rai_net.encodeInput(envState, goalState=goals_tmp)
			if change and rai_net.setup=="minimal" and verbose:
				print("\tGoal changed to "+rai_net.goalString)
		else:
			# Get encoded input
			inputState=rai_net.encodeInput(envState)

		if not cheat_tree:
			# Get high-level action directly through policy
			outEncoded = rai_net.processPrediction(inputState)
			outDecoded = rai_net.decodeAction1(outEncoded)
		else:
			outDecoded=""
		if cheat_tree or skeleton + " " +outDecoded in infeasibleSkeletons+depthSkeletons or (outDecoded in infeasibleSkeletons+depthSkeletons and depth==0) or (outDecoded not in rai_net.lgp.getDecisions() and not rai_net.lgp.getDecisions() == []) or rai_net.checkPlaceSame(outDecoded):
			# if cheat_tree logically or skeleton labeled infeasible or invalid high-level action 
			old=outDecoded
			infeasibleSke=[]
			depthSke=[]
			if depth==0:
				# Find high-level actions that lead to skeletons labeled infeasbile
				for tmpske in infeasibleSkeletons:
					tmpskeStep = rai_world.splitStringStep(tmpske, list_old=[])
					if len(tmpskeStep)==1:
						infeasibleSke.append(tmpskeStep[-1])
				for tmpske in depthSkeletons:
					tmpskeStep = rai_world.splitStringStep(tmpske, list_old=[])
					if len(tmpskeStep)==1:
						depthSke.append(tmpskeStep[-1])

			else:
				# Find high-level actions that lead to skeletons labeled infeasbile
				for tmpske in infeasibleSkeletons:
					tmpskeStep = rai_world.splitStringStep(tmpske, list_old=[])
					if skeleton == ' '.join(tmpskeStep[:-1]):
						infeasibleSke.append(tmpskeStep[-1])
				for tmpske in depthSkeletons:
					tmpskeStep = rai_world.splitStringStep(tmpske, list_old=[])
					if skeleton == ' '.join(tmpskeStep[:-1]):
						depthSke.append(tmpskeStep[-1])

			# Evaluate heuristic for all possible high-level actions
			outDecoded, prob = rai_net.evalPredictions(inputState, infeasible=infeasibleSke, maxdepth=depthSke, prevSke=outDecoded, depth=depth+1, tries=tries)
			if verbose:
				print("MP Decision", outDecoded, "\twith probability", prob)
				print("\tInstead of: ".expandtabs(4), old)
			tmpDes.append("MP")
			tmpDes.append(old)
			tmpDes.append(prob)
		else:
			if verbose:
				print("NN Decision", outDecoded)
			tmpDes.append("NN")

		endtime=time.time()

		timeHP=timeHP+endtime-starttime

		# Append to skeleton
		if skeleton=="":
			skeleton=outDecoded
		else:
			skeleton = skeleton + " " + outDecoded

		if tmpskeleton=="":
			tmpskeleton=outDecoded
		else:
			tmpskeleton = tmpskeleton + " " + outDecoded
		depth+=1

		# Walk to node
		rai_net.lgp.walkToRoot()
		rai_net.lgp.walkToNode(tmpskeleton,0)
		starttime=time.time()
		try:
			# Solve lgp
			komo = rai_world.runLGP(rai_net.lgp, BT.seq, verbose=0, view=False)
			if not planOnly:
				komo = rai_world.runLGP(rai_net.lgp, BT.seqPath, verbose=0, view=False)

		except:
			print("Can not solve komo path for skeleton:", skeleton)
			successmsg="KOMO path failed for goal"
			rai_net.lgp.walkToRoot()
			rai_net.K.copy(K0)
			break
		endtime=time.time()
		timeOP=timeOP+endtime-starttime

		# Determine seq and seqPath feasibility: Currently not used any further
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

		# Copy condiguration from komo solution
		if planOnly:
			komo.getKFromKomo(rai_net.K, komo.getPathFrames(rai_net.logicalNames).shape[0]-2)
		else:
			komo.getKFromKomo(rai_net.K, komo.getPathFrames(rai_net.logicalNames).shape[0]-1)

		if newLGP:
			if rai_net.restartLGP():
				tmpskeleton=""

		# Wait (for vid recording)
		if not (showFinal or rai_net.view):
			time.sleep(waitTime)
		
		# Stop if terminal
		if rai_net.lgp.isTerminal() or len(rai_net.preprocessGoals())==0:
			rai_net.K.copy(K0)
			if newLGP:
				rai_net.restartLGP()
				rai_net.lgp.walkToNode(skeleton,0)
			break
		# Stop if maximum depth reached
		if depth==rai_net.maxDepth:
			print("Maximum depth reached: "+str(rai_net.maxDepth))
			successmsg="Maximum depth of "+str(rai_net.maxDepth)+" reached for goal"
			rai_net.K.copy(K0)
			if newLGP:
				rai_net.restartLGP()
			break

	# Get feasible from final high-level action from seqPath optimization
	if not planOnly:
		feastmp = rai_net.lgp.returnFeasible(BT.seqPath)
	else:
		feastmp = False

	if skeleton=="":
		skeleton="(grasp pr2R obj0) (place pr2R obj0 obj1) (grasp pr2L obj2) (grasp pr2R obj1) (place pr2R obj1 obj2) (grasp pr2R obj3) (place pr2L obj2 obj3) (grasp pr2L obj4) (place pr2R obj3 obj4) (place pr2L obj4 obj5) (grasp pr2R obj6) (place pr2R obj6 obj7) (grasp pr2L obj5) (place pr2L obj5 obj6) (grasp pr2L obj7) (place pr2L obj7 tray) (grasp pr2R obj9) (place pr2R obj9 obj10) (grasp pr2R obj8) (place pr2R obj8 obj9) (grasp pr2R obj11) (place pr2R obj11 obj12) (grasp pr2R obj10) (place pr2R obj10 obj11) (grasp pr2R obj13) (place pr2R obj13 obj14) (grasp pr2L obj12) (place pr2L obj12 obj13) (grasp pr2L obj14) (place pr2L obj14 tray)"
		#skeleton="(grasp pr2R obj0) (place pr2R obj0 obj1) (grasp pr2L obj2) (grasp pr2R obj1) (place pr2R obj1 obj2) (place pr2L obj2 obj3) (grasp pr2L obj4) (place pr2L obj4 obj5) (grasp pr2L obj3) (place pr2L obj3 obj4) (grasp pr2L obj5) (place pr2L obj5 obj6) (grasp pr2L obj7) (place pr2L obj7 tray) (grasp pr2L obj6) (place pr2L obj6 obj7) (grasp pr2L obj9) (place pr2L obj9 obj10) (grasp pr2R obj8) (place pr2R obj8 obj9) (grasp pr2R obj11) (grasp pr2L obj10) (place pr2R obj11 obj12) (place pr2L obj10 obj11) (grasp pr2R obj13) (place pr2R obj13 obj14) (grasp pr2L obj12) (place pr2L obj12 obj13) (grasp pr2L obj14) (place pr2L obj14 tray)"
		rai_net.lgp.walkToRoot()
		rai_net.lgp.walkToNode(skeleton,0)

	if len(rai_net.preprocessGoals())>0:
		if verbose:
			print("--!! Goal not reached !!--")
		feastmp=False
	else:
		if verbose:
			print("\nSkeleton found. Show path and node info")
		successmsg="Successfully reached goal"
		if planOnly:
			# determine feasibility through seqPth optimization
			starttime=time.time()
			try:
				#if newLGP:
				#	for i in range(6):
				#		tmpnoise=0.01-i*0.002
				#		print("----- Noise for seq optimization "+str(tmpnoise)+" -----")
				#		tmptime=time.time()
				#		_ = rai_world.runLGP(rai_net.lgp, BT.seq, verbose=0, view=False, initnoise=tmpnoise)
				#		endtime=time.time()
				#		print("Constraint: seq ",rai_net.lgp.returnConstraint(BT.seq))
				#		print(str(endtime-tmptime)+" sec")
				#		if rai_net.lgp.returnFeasible(BT.seq):
				#			print("seq bound feasible")
				#			break
				#		else:
				#			print(str(i) + ": seq infeasible")
			
				#for i in range(11):
				#	tmpnoise=0.02-i*0.002
				#	print("----- Noise for seq optimization "+str(tmpnoise)+" -----")
				#	tmptime=time.time()
				#	_ = rai_world.runLGP(rai_net.lgp, BT.seqPath, verbose=0, view=showFinal, initnoise=tmpnoise)
				#	endtime=time.time()
				#	print("Constraint: seq ",rai_net.lgp.returnConstraint(BT.seq), ", seqPath ",rai_net.lgp.returnConstraint(BT.seqPath))
				#	print(str(endtime-tmptime)+" sec")
				#	feastmp = rai_net.lgp.returnFeasible(BT.seqPath)
				#	if feastmp:
				#		break
				#	else:
				#		print(str(i) + ": seqPath infeasible")

				for i in [0, -1, 1, -2, 2 , -3, 3]:
					tmpnoise=0.01+i*0.002
					print("----- Noise for seq and seqPath optimization "+str(tmpnoise)+" -----")
					tmptime=time.time()
					if newLGP:
						_ = rai_world.runLGP(rai_net.lgp, BT.seq, verbose=0, view=False, initnoise=tmpnoise)
					_ = rai_world.runLGP(rai_net.lgp, BT.seqPath, verbose=0, view=showFinal, initnoise=tmpnoise)
					endtime=time.time()
					print("Constraint: seq ",rai_net.lgp.returnConstraint(BT.seq), ", seqPath ",rai_net.lgp.returnConstraint(BT.seqPath))
					print(str(endtime-tmptime)+" sec")
					feastmp = rai_net.lgp.returnFeasible(BT.seqPath)
					if feastmp:
						break
					else:
						print(str(i) + ": seqPath infeasible")

			except:
				print("Can not solve komo for skeleton:", skeleton)
				rai_net.K.copy(K0)
				successmsg="KOMO failed for goal"
			endtime=time.time()
			print("\nTime HP", timeHP)
			print("Time OP next config", timeOP)
			print("Time OP full path",endtime-starttime)

	if showFinal:
		rai_net.lgp.nodeInfo()
		
	rai_net.lgp.walkToRoot()
	rai_net.K.copy(K0)

	if not feastmp and verbose:
		print("--!! Infeasible Skeleton !!--")

	return skeleton, typeDecision, successmsg, feastmp

def main():
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
	parser.add_argument('--allEnv', dest='allEnv', action='store_true')
	parser.set_defaults(allEnv=False)
	parser.add_argument('--showFinal', dest='showFinal', action='store_true')
	parser.set_defaults(showFinal=False)
	parser.add_argument('--viewConfig', dest='viewConfig', action='store_true')
	parser.set_defaults(viewConfig=False)
	parser.add_argument('--NoPlanOnly', dest='planOnly', action='store_false')
	parser.add_argument('--planOnly', dest='planOnly', action='store_true')
	parser.set_defaults(planOnly=True)
	parser.add_argument('--newLGP', dest='newLGP', action='store_true')
	parser.set_defaults(newLGP=False)
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
	parser.add_argument('--maxTries', type=int, default=1)
	
	args = parser.parse_args()
	path_rai = args.rai_dir
	verbose=args.verbose
	waitTime= args.waitTime

	model_dir=args.model_dir # model_dir: timestamp of Policy directory or model_dir_data
	model_dir_data=args.model_dir_data # model_dir_data_ timestamp of data set
	if model_dir_data=="":
		model_dir_data=model_dir

	# Training parameters
	train_only = args.train_only #only trains model
	saveModel = args.saveModel #trains model
	
	# Instruction parameters
	epochs_inst=args.epochs_inst
	n_size_inst=args.size_inst
	n_layers_inst=args.hlayers_inst
	n_layers_inst2=args.hlayers_inst2

	# Grasp and Place parameters
	epochs_grasp=args.epochs_grasp
	n_size_grasp=args.size_grasp
	n_layers_grasp=args.hlayers_grasp
	epochs_place=args.epochs_place
	n_size_place=args.size_place
	n_layers_place=args.hlayers_place
	batch_size=args.batch_size

	# Learning parameters
	lr=args.lr
	lr_drop=args.lr_drop
	epoch_drop=args.epoch_drop
	clipnorm=args.clipnorm
	val_split=args.val_split
	reg=args.reg_l2
	reg0=args.reg0_l2

	nenv=args.env
	goalString=args.goal # objective if not (train_only or exclude)
	setup=args.setup
	NNmode=args.NNmode # minimal(impl1) FFnew(impl2) mixed10(impl3)
	dataMode=args.datasetMode # 1(global coord) 2(relative coord) 3(global coord+encoder) 4(relative coord+encoder)
	
	# Testing parameters
	maxDepth=args.maxDepth # maximum depth before reattempting
	maxTries=args.maxTries # maximum tries

	cheat_goalstate= args.cheat_goalstate # adapts goal state if one goal formulation is satisfied
	cheat_tree=args.cheat_tree # always evaluate heuristic for all high-level actions

	completeTesting=args.completeTesting # Test all objectives
	allEnv=args.allEnv # Test a sequence of env
	showFinal=args.showFinal # Display final path
	viewConfig=args.viewConfig # enable K.view
	planOnly=args.planOnly # seq used for next config instead if seqPath
	exclude=args.exclude # Tests only objectives that are excluded in training set. Here: (on red table1)
	newLGP=args.newLGP

	# train_only starts at objective
	start=args.start
	startSub=args.startSub
	start0=start

	if not dataMode in [1,2,3,4]:
		planOnly=False
		NotImplementedError
	
	# Objectives to exclude
	obg = [[],[]]#[[15], [10,20,30,39,48,53,58,63,68]]
	or1 = [[2], [11,12,13,14,15,16,17,18,19,20]]
	og2 = [[],[]]#[[8], [3,13,23,33,42,59,60,61,62,63]]

	#-------------------------------------------------------------------------------------------------------------------------	
	print("Setting up basic Config and FOL for env: "+str(nenv))
	rai=rai_world.RaiWorld(path_rai, nenv, setup, goalString, verbose, maxDepth=maxDepth, NNmode=NNmode, datasetMode=dataMode, view=viewConfig)

	print("\nModel and dataset")
	if saveModel:
		# Train model using data set from model_dir_data
		rai.saveFit(model_dir_data,epochs_inst, n_layers_inst, n_size_inst, epochs_grasp, n_layers_grasp, n_size_grasp, epochs_place, n_layers_place, n_size_place,
					lr, lr_drop, epoch_drop, clipnorm, val_split, reg, reg0, batch_size,n_layers_inst2=n_layers_inst2)
		print("Model trained: "+rai.rai_net.timestamp)
	else:
		# Load model from model_dir
		rai.loadFit(model_dir)
		print("Model loaded")

	if not train_only and not model_dir=="":
		if planOnly:
			append_test="_plan"
		else:
			append_test=""

		if completeTesting or exclude:
			# Select initial configurations to test
			if allEnv:
				if nenv in range(100,104):
					rangeEnv=range(nenv,104)
				elif nenv in range(106,109):
					rangeEnv=range(nenv,109)
					#rangeEnv=range(nenv,110)
				else:
					rangeEnv=[29,38,46,56,64,73,82,91,
							32,39,48,57,65,75,83,93]#,
							#33,40,50,58,66,78,88,94,
							#35,42,51,59,68,80,89,97,
							#43,54,63,72,81,90]
					rangeEnv=rangeEnv[rangeEnv.index(nenv):]
			else:
				rangeEnv=range(nenv,nenv+1)


			for nenv in rangeEnv:
				## env 109 not used for testing anymore
				#if nenv ==109:
				#	rai.maxDepth=rai.maxDepth+2
				#	maxTries=maxTries+2

				summary=[[],[],[],[],[]] #optimal feasible infeasible-op infeasible no

				start=start0
				rai.nenv=nenv
				path=createResults(rai,cheat_tree=cheat_tree, cheat_goalstate=cheat_goalstate, start=start0, planOnly=planOnly)
				if 2*(start-1) < len(minimal_experiment.Sets):
					# Test objectives consisting of 3 goal formulations twice - once for each sequence
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

						# Reload configuration
						rai.redefine(goal, nenv=nenv)

						for tries in range(maxTries):
							# Set objective
							rai.resetFit(cheatGoalState=cheat_goalstate, goal=goal)
							print("----Test Goal "+strgoal+": '"+rai.goalString_orig+"' for env "+str(rai.nenv)+"----\n")
							
							# Find skeleton
							skeleton, typeDecision,successmsg, feasible=buildSkeleton(rai, cheat_tree=cheat_tree, cheat_goalstate=cheat_goalstate, showFinal=showFinal, waitTime=waitTime, planOnly=planOnly, infeasibleSkeletons=infeasibleSkeletons, depthSkeletons=depthSkeletons, tries=tries, newLGP=newLGP)
							
							if successmsg=="Maximum depth of "+str(rai.maxDepth)+" reached for goal":
								# If maximum depth reached: label all partial skeletons infeasible
								depthSkeletons= depthSkeletons + rai_world.splitStringPath(skeleton, list_old=[])
							elif not feasible:
								# If not feasible: label all partial skeletons infeasible
								infeasibleSkeletons= infeasibleSkeletons + rai_world.splitStringPath(skeleton, list_old=[])
							
							# Write results to txt file
							writeResults(rai,skeleton,typeDecision,successmsg,path,goalnumber_string=strgoal, planOnly=planOnly, feasible=feasible, tries=tries)

							if feasible or skeleton=="":
								break
						# Print results to console
						printResult(rai, skeleton)

						if i%2==0:
							# If first sequence checked
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
							# If second sequence checked
							# If same skeleton twice: both feasible if one is feasible
							# Assign to: optimal feasible infeasible-op infeasible no
							if skeleton == skeletonPrev:
								feasible = feasible or idxsol in [0,1]
								if feasible and idxsol in [2,3]:
									idxsol=idxsol-2
							elif not idxsol==4:
								summary[idxsol].append(strgoalPrev)

							if successmsg=="Successfully reached goal":
								solutions, _, _ = minimal_experiment.getData(nenv=nenv, nset=numGoal)
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
									
									if skeleton == skeletonPrev:
										summary[idxsol].append(strgoalPrev)
							else:
								summary[4].append(strgoal)

					start=0
				else:
					numGoal=start-1
					start=(start-1)-len(minimal_experiment.Sets)/2

				for i in range(int(start),len(minimal_experiment.test)):
					# Test goal formulation
					numGoal+=1
					if exclude and not i+1 in or1[0]+og2[0]+obg[0]:
						continue

					strgoal=str(numGoal).zfill(3)+"-1"

					goal= minimal_experiment.test[i]+" "+minimal_experiment.test[i]
					infeasibleSkeletons=[]
					depthSkeletons=[]
					# Reload configuration
					rai.redefine(goal, nenv=nenv)

					for tries in range(maxTries):
						# Set objective
						rai.resetFit(cheatGoalState=cheat_goalstate, goal=goal)
						print("----Test Goal "+strgoal+": '"+rai.goalString_orig+"' for env "+str(rai.nenv)+"----\n")

						skeleton, typeDecision,successmsg, feasible=buildSkeleton(rai,cheat_tree=cheat_tree, cheat_goalstate=cheat_goalstate, showFinal=showFinal,waitTime=waitTime, planOnly=planOnly, infeasibleSkeletons=infeasibleSkeletons, depthSkeletons=depthSkeletons, tries=tries, newLGP=newLGP)
						if successmsg=="Maximum depth of "+str(rai.maxDepth)+" reached for goal":
							# If maximum depth reached: label all partial skeletons infeasible
							depthSkeletons= depthSkeletons + rai_world.splitStringPath(skeleton, list_old=[])
						elif not feasible:
							# If infeasible: label all partial skeletons infeasible
							infeasibleSkeletons= infeasibleSkeletons + rai_world.splitStringPath(skeleton, list_old=[])
						
						# Write results to txt file
						writeResults(rai,skeleton,typeDecision,successmsg,path, goalnumber_string=strgoal, planOnly=planOnly, feasible=feasible)
						if feasible:
							break
					# Print results to console
					printResult(rai, skeleton)

					# Assign to: optimal feasible infeasible-op infeasible no
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
				
				# Write summary
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
			# Test single objective
			print("\nPredict: "+goalString)
			path=createResults(rai,cheat_tree=cheat_tree, cheat_goalstate=cheat_goalstate, start=start0, planOnly=planOnly, test=True)
			infeasibleSkeletons=[]
			depthSkeletons=[]
			teststep=1
			#input("Press enter to continue")
			time.sleep(1)
			starttime=time.time()
			for tries in range(maxTries):
				# Set objective
				rai.resetFit(cheatGoalState=cheat_goalstate, goal=goalString)
				skeleton,typeDecision,successmsg, feasible=buildSkeleton(rai, cheat_tree=cheat_tree, cheat_goalstate=cheat_goalstate, planOnly=planOnly, infeasibleSkeletons=infeasibleSkeletons, depthSkeletons=depthSkeletons, verbose=True, showFinal=showFinal, newLGP=newLGP)

				if successmsg=="Maximum depth of "+str(rai.maxDepth)+" reached for goal":
					# If maximum depth reached: label all partial skeletons infeasible
					depthSkeletons= depthSkeletons + rai_world.splitStringPath(skeleton, list_old=[])
				elif not feasible:
					# If infeasible: label all partial skeletons infeasible
					infeasibleSkeletons= infeasibleSkeletons + rai_world.splitStringPath(skeleton, list_old=[])
				# Write results to txt file
				writeResults(rai,skeleton,typeDecision,successmsg,path,goalnumber_string="", planOnly=planOnly, feasible=feasible, tries=tries, test=True)
				if feasible:
					break
				teststep=teststep+1
			endtime=time.time()
			if not feasible:
				successmsg="Infeasible solution"
			print("time: "+str(endtime-starttime)+", tries: "+str(teststep)+", "+successmsg)
			# Print results to console
			printResult(rai, skeleton)
			if not feasible:
				print("Infeasible Solution")
			if saveModel and successmsg=="Successfully reached goal":
				with open(path_rai+'/logs/toTest.txt', 'a+') as f:
					f.write(rai.model_dir+'\n')

	if not completeTesting and not train_only:
		input("Press Enter to end Program...")



if __name__ == "__main__":
    main()
