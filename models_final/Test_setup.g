# robot adaptations

#Edit worldTranslationRotation { ctrl_H=1 }
Edit worldTranslationRotation { type=transXY } # type=hingeX hingeY hingeZ transX transY transZ transXY trans3

body slider1a { type=box size=[.2 .02 .02 0] color=[.5 .5 .5 .0] }
body slider1b { type=box size=[.2 .02 .02 0] color=[.8 .3 .3 .0] }
joint slider1Joint(slider1a slider1b){ type=transX ctrl_H=.1 }
shape (slider1b){ rel=<T t(.1 0 0)> type=5 size=[.1 .1 .1] color=[0 1 0] }


# tables
Frame table1{ shape:ssBox, X=<T t(0.1 1.1 .6) d(90 0 0 1)>, size=[.8 2.2 .1 .02], color=[.3 .3 .3] fixed, contact, logical={ table } }
#Frame table2{ shape:ssBox, X=<T t(.8 0 .6)>, size=[.8 1.4 .1 .02], color=[.3 .3 .3] fixed, contact, logical={ table } }
Frame table2{ shape:ssBox, X=<T t(1.4 -0.3 .6)>, size=[1.8 1.8 .1 .02], color=[.3 .3 .3] fixed, contact, logical={ table } }

Frame table3{ shape:ssBox, X=<T t(.1 -1.6 .6) d(90 0 0 1)>, size=[0.1 2.4 .1 .02], color=[.3 .3 .3] fixed, contact, logical={} }

Frame table4{ shape:ssBox, X=<T t(-1.4 -0.3 .6)>, size=[0.1 2.0 .1 .02], color=[.3 .3 .3] fixed, contact, logical={} }


# objects in Test_setup_<number>

### hook

body nostick { type=5 size=[.2 .2 .2] }
joint (table2 nostick) { from=<T t(0 0 .02) t(-.6 -.7 .05)> type=rigid }
shape stick(nostick) { type=ssBox size=[.8 .025 .04 .01] color=[.6 .3 0] contact, logical={ object } }
shape stickTip (nostick) { rel=<T t(.4 .1 0) d(90 0 0 1)> type=ssBox size=[.2 .026 .04 0.01] color=[.6 .3 0], logical={ object, pusher } }

