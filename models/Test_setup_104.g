Include = 'Test_setup.g'

#env3
body red {type:ssBox size:[.15 .15 .15 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint0(table2 red){type=rigid, Q:<t(-0.1 0.5 .15)>}
body green {type:ssBox size:[.15 .15 .15 .02] color:[0. 1. 0.], contact, logical={ object, table }}
joint joint1 (table1 green) {type=rigid, Q:<t(-0.1 -0.15 .15)>}
body blue {type:ssBox size:[.15 .15 .15 .02] color:[0. 0. 1.], contact, logical={ object, table }}
joint joint2(table2 blue) {joint:rigid, Q:<t(0.05 0.15 .15)>}
body yellow {type:ssBox size:[.15 .15 .15 .02] color:[1. 1. 0.], contact, logical={ object, table }}
joint joint3(table2 yellow) {joint:rigid, Q:<t(0.05 -0.4 .15)>}
body cyan {type:ssBox size:[.15 .15 .15 .02] color:[0. 1. 1.], contact, logical={ object, table }}
joint joint4 (table1 cyan) {type=rigid, Q:<t(-0.1 0.35 .15)>}
