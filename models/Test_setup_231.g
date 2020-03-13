Include = 'Test_setupStack.g'

#env202
body red {type:ssBox size:[.15 .15 .15 .02] color:[1. 0. 0.], contact, logical={ object, table }}

body green {type:ssBox size:[.15 .15 .15 .02] color:[0. 1. 0.], contact, logical={ object, table }}

body blue {type:ssBox size:[.15 .15 .15 .02] color:[0. 0. 1.], contact, logical={ object, table }}


joint joint0(table1 red){type=rigid, Q:<t(0.1 0.2 .15)>}
joint joint1 (red green) {type=rigid, Q:<t(-0.05 0.045 .15)>}
joint joint2(green blue) {joint:rigid, Q:<t(0.45 -.45 .15)>}
