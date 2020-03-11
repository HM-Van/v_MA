Include = 'Test_setup.g'

#env3
body red {type:ssBox size:[.15 .15 .15 .02] color:[1. 0. 0.], contact, logical={ object , table}}
body green {type:ssBox size:[.15 .15 .15 .02] color:[0. 1. 0.], contact, logical={ object, table }}
body blue {type:ssBox size:[.15 .15 .15 .02] color:[0. 0. 1.], contact, logical={ object, table }}
body cyan {type:ssBox size:[.15 .15 .15 .02] color:[0. 1. 1.], contact, logical={ object, table }}
body orange {type:ssBox size:[.15 .15 .15 .02] color:[1. .5 0.], contact, logical={ object, table }}
body yellow {type:ssBox size:[.15 .15 .15 .02] color:[1. 1. 0.], contact, logical={ object, table }}

joint joint0(table1 green){type=rigid, Q:<t(-0.1 0.5 .15)>}
joint joint1 (table1 yellow) {type=rigid, Q:<t(-0.3 -0.5 .15)>}
joint joint2(table1 cyan) {joint:rigid, Q:<t(-0.05 -0.05 .15)>}
joint joint3 (green red) {type=rigid, Q:<t(0.07 -0.05 .15)>}
joint joint4 (yellow blue) {type=rigid, Q:<t(0.07 -0.05 .15)>}
joint joint5 (cyan orange) {type=rigid, Q:<t(0.07 -0.05 .15)>}

body tray{type:ssBox size:[.7 .7 .04 .02] color:[0.7 .7 0.7], contact, logical={ object, table }}
joint jointTray(table2 tray) {joint:rigid, Q:<t(0.0 -0.2 .07)>}
