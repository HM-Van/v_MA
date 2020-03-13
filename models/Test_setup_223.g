Include = 'Test_setupStack.g'

#env203
body red {type:ssBox size:[.15 .15 .15 .02] color:[1. 0. 0.], contact, logical={ object, table }}

body green {type:ssBox size:[.15 .15 .15 .02] color:[0. 1. 0.], contact, logical={ object, table }}

body blue {type:ssBox size:[.15 .15 .15 .02] color:[0. 0. 1.], contact, logical={ object, table }}


joint joint0(green red){type=rigid, Q:<t(0.02 0.04 .15)>}
joint joint1 (table1 green) {type=rigid, Q:<t(0.05 -0.1 .15)>}
joint joint2(table1 blue) {joint:rigid, Q:<t(0.25 .2 .15)>}
