Include = 'Test_setup.g'

#env3
body red {type:ssBox size:[.15 .15 .15 .02] color:[1. 0. 0.], contact, logical={ object, table }}
body green {type:ssBox size:[.15 .15 .15 .02] color:[0. 1. 0.], contact, logical={ object, table }}
body blue {type:ssBox size:[.15 .15 .15 .02] color:[0. 0. 1.], contact, logical={ object, table }}


joint joint0(blue red){type=rigid, Q:<t(0.07 -0.07 .175)>}
joint joint1 (table2 green) {type=rigid, Q:<t(-0.3 0.35 .15)>}
joint joint3 (table2 blue) {type=rigid, Q:<t(0.1 -0.1 .15)>}
