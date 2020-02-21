Include = 'keywords.g'
#Include = 'kin-stickHandover.g'

FOL_World{
  hasWait=false
  gamma = 1.
  stepCost = 1.
  timeCost = 0.
}

## activities
grasping
placing
handing
attaching
pushing

## basic predicates
gripper
object
table
wall
attachable
pusher
partOf
world

on
ontop
busy     # involved in an ongoing (durative) activity
animate  # the object is in principle movable
free     # gripper hand is free
held     # object is held by an gripper
#notheld  # triggers the rule to set !held for all parts
grasped  # gripper X holds/has grasped object Y
placed   # gripper X holds/has grasped object Y
attached
hasScrew # gripper X holds a screw (screws are not objects/constrants, just a predicate of having a screw...)
fixed    # object X and Y are fixed together
never    # (for debugging)

## KOMO symbols
inside
above
lift
notAbove

touch
impulse
stable
stableOn
dynamic
dynamicOn
liftDownUp

fricSlide
dynamicVert
flagClear

## constants (added by the code)
obj0
obj1
obj2
obj3
tray
stick
stickTip
redBall
blueBall
bucket

red
green
blue
table1
table2

## initial state (generated by the code)
START_STATE {}

### RULES

#####################################################################

#termination rule
#Rule {
#  { (on obj0 tray) (on obj1 tray) (on obj2 tray) }
#  { (QUIT) }
#}

### Reward
REWARD {
#  tree{
#    leaf{ { (grasped handR screwdriverHandle) }, r=10. }
#    weight=1.
#  }
}

#####################################################################

DecisionRule grasp {
  X, Y
  { (gripper X) (object Y) (busy X)! (held Y)! (INFEASIBLE grasp X Y)!}
  { (above Y ANY)! (on ANY Y)! (ontop ANY Y)! (stableOn ANY Y)! 
    (grasped X Y) (held Y) (busy X) # these are only on the logical side, to enable correct preconditions
    (touch X Y) (stable X Y) # these are predicates that enter the NLP
  }
}

#####################################################################
#changed to remove notheld predicate
DecisionRule place {
  X, Y, Z,
  { (grasped X Y) (table Z) (held Y) (on Y Z)! (ontop Y Z)!}
  { (grasped X Y)! (busy X)! (busy Y)! (held Y)! # logic only
    (stable ANY Y)! (touch X Y)! # NLP predicates
    (on Z Y) (above Y Z) (stableOn Z Y) tmp(touch X Y) tmp(touch Y Z)
    (INFEASIBLE grasp ANY Y)! block(INFEASIBLE grasp ANY Y)
  }
}

DecisionRule place {
  X, Y, Z,
  { (gripper X) (table Z) (table X)! (on X Y) (object Y) }
  { (grasped X Y)! (busy X)! (busy Y)! (held Y)! # logic only
    (stable ANY Y)! (touch X Y)! # NLP predicates
    (on Y Z) (above Y Z) (stableOn Z Y) tmp(touch X Y) tmp(touch Y Z)
    (INFEASIBLE grasp ANY Y)! block(INFEASIBLE grasp ANY Y)
  }
}

#####################################################################

Rule{
  A, B
  { (gripper A) (object B) (on A B)  }
  { (on A B)! (grasped A B) (held B) (busy A)}
}

Rule{
  A, B, C
  { (on A B) (on B C) (object A) }
  { (ontop A C)}
}

Rule{
  A, B, C
  { (ontop A B) (on B C) (object A) }
  { (ontop A C)}
}
# This rule seems to break something. But rule below should be sufficient
#Rule{
#  A, B, C
#  { (ontop A C) (on B C) (on A B)! }
#  { (ontop A C)!}
#}

Rule{
  A, B, C
  { (ontop A C) (on B C) (ontop A B)! (on A B)!}
  { (ontop A C)!}
}
