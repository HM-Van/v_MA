Include = 'Test_setup.g'

#env75
body red {type:ssBox size:[.15 .15 .15 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint0(table2 red){type=rigid, Q:<t(-0.25 -0.5 .15)>}
body green {type:ssBox size:[.15 .15 .15 .02] color:[0. 1. 0.], contact, logical={ object, table }}
joint joint1 (table1 green) {type=rigid, Q:<t(0.2 0.1 .15)>}
body blue {type:ssBox size:[.15 .15 .15 .02] color:[0. 0. 1.], contact, logical={ object, table }}
joint joint2(table1 blue) {joint:rigid, Q:<t(-0.2 .55 .15)>}
