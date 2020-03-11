# robot adaptations

#Edit worldTranslationRotation { ctrl_H=1 }
Edit worldTranslationRotation { type=transXY } # type=hingeX hingeY hingeZ transX transY transZ transXY trans3


# tables
#Frame table1{ shape:ssBox, X=<T t(0.1 1.1 .6) d(90 0 0 1)>, size=[.8 2.2 .1 .02], color=[.3 .3 .3] fixed, contact, logical={ table } }

Frame table1{ shape:ssBox, X=<T t(.8 0 .6)>, size=[.8 1.4 .1 .02], color=[.3 .3 .3] fixed, contact, logical={ table } }

# objects in Test_setup_<number>

