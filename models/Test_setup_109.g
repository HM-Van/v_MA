Include = 'Test_setup.g'

#env3
body red {type:ssBox size:[.15 .15 .15 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint0(table1 red){type=rigid, Q:<t(-0.1 0.3 .15)>}
body green {type:ssBox size:[.15 .15 .15 .02] color:[0. 1. 0.], contact, logical={ object, table }}
#joint joint1 (red green) {type=rigid, Q:<t(0.07 -0.07 .175)>}
joint joint1 (red green) {type=rigid, Q:<t(0.0 -0.08 .175)>}
body blue {type:ssBox size:[.15 .15 .15 .02] color:[0. 0. 1.], contact, logical={ object, table }}
joint joint3 (green blue) {type=rigid, Q:<t(-0.08 0.0 .175)>}

body tray{type:ssBox size:[.7 .7 .04 .02] color:[0.7 .7 0.7], contact, logical={ object, table }}
joint jointtray(table2 tray) {joint:rigid, Q:<t(0.0 0.0 .07)>}
