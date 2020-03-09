Include = 'Test_setup0.g'

#(on obj0 obj1) (on obj1 obj2) (on obj2 obj3) (on obj3 obj4) (on obj4 obj5) (on obj5 obj6) (on obj6 obj7) (on obj7 tray) (on obj8 obj9) (on obj9 obj10) (on obj10 obj11) (on obj11 obj12) (on obj12 obj13) (on obj13 obj14) (on obj14 tray)
#(on obj15 obj16) (on obj16 obj17) (on obj17 obj18) (on obj18 obj19) (on obj19 obj20) (on obj20 obj21) (on obj21 obj22) (on obj22 tray) (on obj23 obj24) (on obj24 obj25) (on obj25 obj26) (on obj26 obj27) (on obj27 obj28) (on obj28 obj29) (on obj29 tray)


#env3
body obj0 {type:ssBox size:[.15 .15 .1 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint0(table1 obj0){type=rigid, Q:<t(-0.2 1.1 .125)>}

body obj1 {type:ssBox size:[.15 .15 .1 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint1(table1 obj1){type=rigid, Q:<t(-0.2 0.8 .125)>}

body obj2 {type:ssBox size:[.15 .15 .1 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint2(table1 obj2){type=rigid, Q:<t(-0.2 0.5 .125)>}

body obj3 {type:ssBox size:[.15 .15 .1 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint3(table1 obj3){type=rigid, Q:<t(-0.2 0.2 .125)>}

body obj4 {type:ssBox size:[.15 .15 .1 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint4(table1 obj4){type=rigid, Q:<t(-0.2 -0.1 .125)>}

body obj5 {type:ssBox size:[.15 .15 .1 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint5(table1 obj5){type=rigid, Q:<t(-0.2 -0.4 .125)>}

body obj6 {type:ssBox size:[.15 .15 .1 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint6(table1 obj6){type=rigid, Q:<t(-0.2 -0.7 .125)>}

body obj7 {type:ssBox size:[.15 .15 .1 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint7(table1 obj7){type=rigid, Q:<t(-0.2 -1. .125)>}

body obj8 {type:ssBox size:[.15 .15 .1 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint8(table1 obj8){type=rigid, Q:<t(0.1 0.95 .125)>}

body obj9 {type:ssBox size:[.15 .15 .1 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint9(table1 obj9){type=rigid, Q:<t(0.1 0.65 .125)>}

body obj10 {type:ssBox size:[.15 .15 .1 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint10(table1 obj10){type=rigid, Q:<t(0.1 0.35 .125)>}

body obj11 {type:ssBox size:[.15 .15 .1 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint11(table1 obj11){type=rigid, Q:<t(0.1 0.05 .125)>}

body obj12 {type:ssBox size:[.15 .15 .1 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint12(table1 obj12){type=rigid, Q:<t(0.1 -0.25 .125)>}

body obj13 {type:ssBox size:[.15 .15 .1 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint13(table1 obj13){type=rigid, Q:<t(0.1 -0.55 .125)>}

body obj14 {type:ssBox size:[.15 .15 .1 .02] color:[1. 0. 0.], contact, logical={ object, table }}
joint joint14(table1 obj14){type=rigid, Q:<t(0.1 -0.85 .125)>}

#body obj15 {type:ssBox size:[.15 .15 .1 .02] color:[1. 0. 0.], contact, logical={ object, table }}
#joint joint15(table2 obj15){type=rigid, Q:<t(-0.1 0.8 .125)>}

#body obj16 {type:ssBox size:[.15 .15 .1 .02] color:[1. 0. 0.], contact, logical={ object, table }}
#joint joint16(table2 obj16){type=rigid, Q:<t(-0.1 0.5 .125)>}

#body obj17 {type:ssBox size:[.15 .15 .1 .02] color:[1. 0. 0.], contact, logical={ object, table }}
#joint joint17(table2 obj17){type=rigid, Q:<t(-0.1 -0.5 .125)>}

#body obj18 {type:ssBox size:[.15 .15 .1 .02] color:[1. 0. 0.], contact, logical={ object, table }}
#joint joint18(table2 obj18){type=rigid, Q:<t(-0.1 -0.8 .125)>}




body tray{type:ssBox size:[.75 .75 .04 .02] color:[0.7 .7 0.7], contact, logical={ object, table }}
joint jointtray(table2 tray) {joint:rigid, Q:<t(0.0 0.0 .07)>}
