Include = 'Test_setup.g'

#env70
body red {type:ssBox size:[.15 .15 .15 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint0(table2 red){type=rigid, Q:<t(-0.25 -0.4 .15)>}
body green {type:ssBox size:[.15 .15 .15 .02] color:[0. 1. 0.], contact, logical={ object, table }}
joint joint1 (table2 green) {type=rigid, Q:<t(0.2 -0.5 .15)>}
body blue {type:ssBox size:[.15 .15 .15 .02] color:[0. 0. 1.], contact, logical={ object, table }}
joint joint2(table1 blue) {joint:rigid, Q:<t(0.15 .5 .15)>}
