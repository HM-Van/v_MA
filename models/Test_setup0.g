# robot adaptations

Edit worldTranslationRotation { ctrl_H=1 }
#Edit worldTranslationRotation { type=transXY, ctrl_H=1 } # type=hingeX hingeY hingeZ transX transY transZ transXY trans3


# tables
#Frame table1{ shape:ssBox, X=<T t(0.1 1.1 .5) d(90 0 0 1)>, size=[.8 2.2 .1 .02], color=[.3 .3 .3] fixed, contact, logical={ table } }
##Frame table1{ shape:ssBox, X=<T t(-0.4 1.1 .5) d(90 0 0 1)>, size=[.8 3.2 .1 .02], color=[.3 .3 .3] fixed, contact, logical={ table } }
Frame table1{ shape:ssBox, X=<T t(0.3 1.5 .5) d(90 0 0 1)>, size=[.8 2.6 .1 .02], color=[.3 .3 .3] fixed, contact, logical={ table } }
#Frame table1{ shape:ssBox, X=<T t(0.1 1.1 .5) d(90 0 0 1)>, size=[.8 2.6 .1 .02], color=[.3 .3 .3] fixed, contact, logical={ table } }



#Frame table2{ shape:ssBox, X=<T t(.8 -0.5 .5)>, size=[.8 2.4 .1 .02], color=[.3 .3 .3] fixed, contact, logical={ table } }
Frame table2{ shape:ssBox, X=<T t(1.1 -0.8 .5)>, size=[.8 0.8 .1 .02], color=[.3 .3 .3] fixed, contact, logical={ table } }
#Frame table2{ shape:ssBox, X=<T t(.8 -0.8 .5)>, size=[.8 0.8 .1 .02], color=[.3 .3 .3] fixed, contact, logical={ table } }
# objects in Test_setup_<number>
