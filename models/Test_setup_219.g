Include = 'Test_setupStack.g'

#env203
body red {type:ssBox size:[.15 .15 .15 .02] color:[1. 0. 0.], contact, logical={ object, table }}

body green {type:ssBox size:[.15 .15 .15 .02] color:[0. 1. 0.], contact, logical={ object, table }}

body blue {type:ssBox size:[.15 .15 .15 .02] color:[0. 0. 1.], contact, logical={ object, table }}


joint joint0(table1 red){type=rigid, Q:<t(0.1 0.2 .15)>}
joint joint1 (blue green) {type=rigid, Q:<t(-0.05 0.065 .15)>}
joint joint2(table1 blue) {joint:rigid, Q:<t(0.25 .6 .15)>}
