Include = 'tables.g'

body obj0 {type:ssBox size:[.1 .1 .2 .02] color:[1. 0. 0.], contact, logical={ object }}
joint joint0(table1 obj0){type=rigid, Q:<t(0 0 .15)>}

body obj1 {type:ssBox size:[.1 .1 .2 .02] color:[1. 0. 0.], contact, logical={ object }}
joint joint1 (table1 obj1) {type=rigid, Q:<t(0 .2 .15)>}

body obj2 {type:ssBox size:[.1 .1 .2 .02] color:[1. 0. 0.], contact, logical={ object }}
joint joint2(table1 obj2) {joint:rigid, Q:<t(0 .4 .15)>}

body obj3 {type:ssBox size:[.1 .1 .2 .02] color:[1. 0. 0.], contact, logical={ object }}
joint joint3 (table1 obj3) {joint:rigid, Q:<t(0 .6.15)>}

body tray {type:ssBox size:[.15 .15 .04 .02] color:[0. 1. 0.], logical={ table }}
joint joint4 (table2 tray) {Q:<t(0 0 .07)>}

body tray2 {type:ssBox size:[.27 .27 .04 .02] color:[0. 1. 0.]}
joint joint4 (tray tray2) {Q:<t(0 0 0)>}

# if nostick part of goal: add stick/stickTip to FOL
#body nostick { type=5 size=[.2 .2 .2] }
#joint (table3 nostick) { from=<T t(0 0 .02) t(.1 -.7 .05)> type=rigid }
#shape stick(nostick) { type=ssBox size=[.8 .025 .04 .01] color=[.6 .3 0] contact, logical={ object } }
#shape stickTip (nostick) { rel=<T t(.4 .1 0) d(90 0 0 1)> type=ssBox size=[.2 .026 .04 0.01] color=[.6 .3 0], logical={ object, pusher } }
