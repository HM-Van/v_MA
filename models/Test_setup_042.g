Include = 'Test_setup.g'

#env42
body red {type:ssBox size:[.15 .15 .15 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint0(table1 red){type=rigid, Q:<t(0.2 0.3 .15)>}
body green {type:ssBox size:[.15 .15 .15 .02] color:[0. 1. 0.], contact, logical={ object, table }}
joint joint1 (table1 green) {type=rigid, Q:<t(-0.05 0.35 .15)>}
body blue {type:ssBox size:[.15 .15 .15 .02] color:[0. 0. 1.], contact, logical={ object, table }}
joint joint2(table2 blue) {joint:rigid, Q:<t(-0.15 -.1 .15)>}
