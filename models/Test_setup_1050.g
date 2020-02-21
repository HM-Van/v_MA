Include = 'Test_setup.g'

#env3
body red {type:ssBox size:[.15 .15 .15 .02] color:[1. 0. 0.], contact, logical={ object , table}}
joint joint0(table1 red){type=rigid, Q:<t(-0.1 0.5 .15)>}
body green {type:ssBox size:[.15 .15 .15 .02] color:[0. 1. 0.], contact, logical={ object, table }}
joint joint1 (table1 green) {type=rigid, Q:<t(-0.3 -0.55 .15)>}
body blue {type:ssBox size:[.15 .15 .15 .02] color:[0. 0. 1.], contact, logical={ object, table }}
joint joint2(table1 blue) {joint:rigid, Q:<t(0.05 0.25 .15)>}
#body cyan {type:ssBox size:[.15 .15 .15 .02] color:[0. 1. 1.], contact, logical={ object, table }}
#joint joint3 (table1 cyan) {type=rigid, Q:<t(0.1 -0.25 .15)>}

body tray{type:ssBox size:[.27 .27 .04 .02] color:[0.7 .7 0.7], contact, logical={ object, table }}
joint joint4(table2 tray) {joint:rigid, Q:<t(0.0 0.0 .07)>}
