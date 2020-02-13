import numpy as np

envInfo=x = np.array([[1, 1, 2], #1
                        [2, 2, 1],#2
                        [2, 1, 1],#3
                        [1, 1, 1],#4
                        [1, 1, 1],#5
                        [1, 2, 2],#6
                        [1, 1, 2],#7
                        [2, 2, 1],#8
                        [1, 2, 1],#9
                        [2, 1, 2],#10
                        [2, 2, 2],#11
                        [2, 2, 1],#12
                        [2, 1, 2],#13
                        [1, 2, 2],#14
                        [1, 1, 2],#15
                        [2, 1, 1],#16
                        [1, 2, 1],#17
                        [1, 1, 1],#18
                        [2, 2, 2],#19
                        [2, 1, 1],#20
                        [2, 2, 1],#21
                        [2, 1, 2],
                        [2, 1, 1],
                        [1, 2, 1],
                        [1, 2, 2],
                        [1, 2, 2],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 2],
                        [1, 1, 2],
                        [1, 1, 2],
                        [1, 1, 2],
                        [1, 1, 2],
                        [1, 1, 2],
                        [1, 1, 2],
                        [1, 1, 2],
                        [1, 1, 2],
                        [1, 2, 2],
                        [1, 2, 2],
                        [1, 2, 2],
                        [1, 2, 2],
                        [1, 2, 2],
                        [1, 2, 2],
                        [1, 2, 2],
                        [1, 2, 2],
                        [1, 2, 2],
                        [1, 2, 1],
                        [1, 2, 1],
                        [1, 2, 1],
                        [1, 2, 1],
                        [1, 2, 1],
                        [1, 2, 1],
                        [1, 2, 1],
                        [1, 2, 1],
                        [1, 2, 1],
                        [2, 2, 1],
                        [2, 2, 1],
                        [2, 2, 1],
                        [2, 2, 1],
                        [2, 2, 1],
                        [2, 2, 1],
                        [2, 2, 1],
                        [2, 2, 1],
                        [2, 2, 1],
                        [2, 1, 1],
                        [2, 1, 1],
                        [2, 1, 1],
                        [2, 1, 1],
                        [2, 1, 1],
                        [2, 1, 1],
                        [2, 1, 1],
                        [2, 1, 1],
                        [2, 1, 1],
                        [2, 1, 2],
                        [2, 1, 2],
                        [2, 1, 2],
                        [2, 1, 2],
                        [2, 1, 2],
                        [2, 1, 2],
                        [2, 1, 2],
                        [2, 1, 2],
                        [2, 1, 2],
                        [2, 2, 2],
                        [2, 2, 2],
                        [2, 2, 2],
                        [2, 2, 2],
                        [2, 2, 2],
                        [2, 2, 2],
                        [2, 2, 2],
                        [2, 2, 2],
                        [2, 2, 2],#99
                        [2, 2, 2], #100
                        [2, 1, 1], #101
                        [1, 2, 1], #102
                        [2, 1, 2], #103
                        [2, 1, 2], #104
                        [0, 0, 0], #105
                        [1, 1, 0], #106
                        [2, 0, 1],
                        [0, 2, 2],
                        [1, 0, 0]
                        ],
                        np.int16)

test=["(held red)", "(on red table1)", "(on red table2)", "(on red green)", "(on red blue)",
        "(held green)", "(on green table1)", "(on green table2)", "(on green red)", "(on green blue)",
        "(held blue)", "(on blue table1)", "(on blue table2)", "(on blue red)", "(on blue green)"]

numSets=72
numEnv=21

Sets=[
    "(held red) (held green)", "(held green) (held red)", #1
    "(held red) (on green table1)", "(on green table1) (held red)", #2
    "(held red) (on green table2)", "(on green table2) (held red)", #3
    "(held red) (on green red)", "(on green red) (held red)", #4
    "(held red) (on green blue)", "(on green blue) (held red)", #5
    "(held red) (held blue)", "(held blue) (held red)", #6
    "(held red) (on blue table1)", "(on blue table1) (held red)", #7
    "(held red) (on blue table2)", "(on blue table2) (held red)", #8
    "(held red) (on blue red)", "(on blue red) (held red)", #9
    "(held red) (on blue green)", "(on blue green) (held red)", #10
    "(on red table1) (held green)", "(held green) (on red table1)", #11
    "(on red table1) (on green table1)", "(on green table1) (on red table1)", #12
    "(on red table1) (on green table2)", "(on green table2) (on red table1)", #13
    "(on red table1) (on green red)", "(on green red) (on red table1)", #14
    "(on red table1) (on green blue)", "(on green blue) (on red table1)", #15
    "(on red table1) (held blue)", "(held blue) (on red table1)", #16
    "(on red table1) (on blue table1)", "(on blue table1) (on red table1)", #17
    "(on red table1) (on blue table2)", "(on blue table2) (on red table1)", #18
    "(on red table1) (on blue red)", "(on blue red) (on red table1)", #19
    "(on red table1) (on blue green)", "(on blue green) (on red table1)", #20
    "(on red table2) (held green)", "(held green) (on red table2)", #21
    "(on red table2) (on green table1)", "(on green table1) (on red table2)", #22
    "(on red table2) (on green table2)", "(on green table2) (on red table2)", #23
    "(on red table2) (on green red)", "(on green red) (on red table2)", #24
    "(on red table2) (on green blue)", "(on green blue) (on red table2)", #25
    "(on red table2) (held blue)", "(held blue) (on red table2)", #26
    "(on red table2) (on blue table1)", "(on blue table1) (on red table2)", #27
    "(on red table2) (on blue table2)", "(on blue table2) (on red table2)", #28
    "(on red table2) (on blue red)", "(on blue red) (on red table2)", #29
    "(on red table2) (on blue green)", "(on blue green) (on red table2)", #30
    "(on red green) (held green)", "(held green) (on red green)", #31
    "(on red green) (on green table1)", "(on green table1) (on red green)", #32
    "(on red green) (on green table2)", "(on green table2) (on red green)", #33
    "(on red green) (on green blue)", "(on green blue) (on red green)", #34
    "(on red green) (held blue)", "(held blue) (on red green)", #35
    "(on red green) (on blue table1)", "(on blue table1) (on red green)", #36
    "(on red green) (on blue table2)", "(on blue table2) (on red green)", #37
    "(on red green) (on blue red)", "(on blue red) (on red green)", #38
    "(on red green) (on blue green)", "(on blue green) (on red green)", #39
    "(on red blue) (held green)", "(held green) (on red blue)", #40
    "(on red blue) (on green table1)", "(on green table1) (on red blue)", #41
    "(on red blue) (on green table2)", "(on green table2) (on red blue)", #42
    "(on red blue) (on green red)", "(on green red) (on red blue)", #43
    "(on red blue) (on green blue)", "(on green blue) (on red blue)", #44
    "(on red blue) (held blue)", "(held blue) (on red blue)", #45
    "(on red blue) (on blue table1)", "(on blue table1) (on red blue)", #46
    "(on red blue) (on blue table2)", "(on blue table2) (on red blue)", #47
    "(on red blue) (on blue green)", "(on blue green) (on red blue)", #48
    "(held green) (held blue)", "(held blue) (held green)", #49
    "(held green) (on blue table1)", "(on blue table1) (held green)", #50
    "(held green) (on blue table2)", "(on blue table2) (held green)", #51
    "(held green) (on blue red)", "(on blue red) (held green)", #52
    "(held green) (on blue green)", "(on blue green) (held green)", #53
    "(on green table1) (held blue)", "(held blue) (on green table1)", #54
    "(on green table1) (on blue table1)", "(on blue table1) (on green table1)", #55
    "(on green table1) (on blue table2)", "(on blue table2) (on green table1)", #56
    "(on green table1) (on blue red)", "(on blue red) (on green table1)", #57
    "(on green table1) (on blue green)", "(on blue green) (on green table1)", #58
    "(on green table2) (held blue)", "(held blue) (on green table2)", #59
    "(on green table2) (on blue table1)", "(on blue table1) (on green table2)", #60
    "(on green table2) (on blue table2)", "(on blue table2) (on green table2)", #61
    "(on green table2) (on blue red)", "(on blue red) (on green table2)", #62
    "(on green table2) (on blue green)", "(on blue green) (on green table2)", #63
    "(on green red) (held blue)", "(held blue) (on green red)", #64
    "(on green red) (on blue table1)", "(on blue table1) (on green red)", #65
    "(on green red) (on blue table2)", "(on blue table2) (on green red)", #66
    "(on green red) (on blue red)", "(on blue red) (on green red)", #67
    "(on green red) (on blue green)", "(on blue green) (on green red)", #68
    "(on green blue) (held blue)", "(held blue) (on green blue)", #69
    "(on green blue) (on blue table1)", "(on blue table1) (on green blue)", #70
    "(on green blue) (on blue table2)", "(on blue table2) (on green blue)", #71
    "(on green blue) (on blue red)", "(on blue red) (on green blue)", #72
]

def getEnvInfo(env,key):
    if key=="r" or key=="red" or key==0 or key=="0":
        return envInfo[env-1,0]
    if key=="g" or key=="green" or key==1 or key=="1":
        return envInfo[env-1,1]
    if key=="b" or key=="blue" or key==2 or key=="2":
        return envInfo[env-1,2]
    else:
        input("Not defined: ", key)
        return 0

#test=["(held red)", "(on red table1)", "(on red table2)", "(on red green)", "(on red blue)",
#        "(held green)", "(on green table1)", "(on green table2)", "(on green red)", "(on green blue)",
#        "(held blue)", "(on blue table1)", "(on blue table2)", "(on blue red)", "(on blue green)"]
def getData1(nset=1, nenv=1):
    solutions=[]
    numLoops=0

    if nset==1:
        goalString=["(held red) (held red)"]
        solutions=["(grasp pr2R red)",
                    "(grasp pr2L red)"]
        numLoops=1*2*1
    
    elif nset==2:
        goalString=["(on red table1) (on red table1)"]
        if getEnvInfo(nenv, "red") in [0,2]:
            solutions=["(grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L red) (place pr2L red table1)"]
            numLoops=1*2*2
        elif getEnvInfo(nenv,"red") in [0,1]:
            solutions=[]
            numLoops=0

    elif nset==3:
        goalString=["(on red table2) (on red table2)"]
        if getEnvInfo(nenv, "red") in [0,1]:
            solutions=["(grasp pr2R red) (place pr2R red table2)",
                        "(grasp pr2L red) (place pr2L red table2)"]
            numLoops=1*2*2
        elif getEnvInfo(nenv,"red") in [0,2]:
            solutions=[]
            numLoops=0

    elif nset==4:
        goalString=["(on red green) (on red green)"]
        solutions=["(grasp pr2R red) (place pr2R red green)",
                    "(grasp pr2L red) (place pr2L red green)"]
        numLoops=1*2*2

    elif nset==5:
        goalString=["(on red blue) (on red blue)"]
        solutions=["(grasp pr2R red) (place pr2R red blue)",
                    "(grasp pr2L red) (place pr2L red blue)"]
        numLoops=1*2*2

    elif nset==6:
        goalString=["(held green) (held green)"]
        solutions=["(grasp pr2R green)",
                    "(grasp pr2L green)"]
        numLoops=1*2*1
    
    elif nset==7:
        goalString=["(on green table1) (on green table1)"]
        if getEnvInfo(nenv, "green") in [0,2]:
            solutions=["(grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L green) (place pr2L green table1)"]
            numLoops=1*2*2
        elif getEnvInfo(nenv,"green") in [0,1]:
            solutions=[]
            numLoops=0

    elif nset==8:
        goalString=["(on green table2) (on green table2)"]
        if getEnvInfo(nenv, "green") in [0,1]:
            solutions=["(grasp pr2R green) (place pr2R green table2)",
                        "(grasp pr2L green) (place pr2L green table2)"]
            numLoops=1*2*2
        elif getEnvInfo(nenv,"green") in [0,2]:
            solutions=[]
            numLoops=0

    elif nset==9:
        goalString=["(on green red) (on green red)"]
        solutions=["(grasp pr2R green) (place pr2R green red)",
                    "(grasp pr2L green) (place pr2L green red)"]
        numLoops=1*2*2

    elif nset==10:
        goalString=["(on green blue) (on green blue)"]
        solutions=["(grasp pr2R green) (place pr2R green blue)",
                    "(grasp pr2L green) (place pr2L green blue)"]
        numLoops=1*2*2

    elif nset==11:
        goalString=["(held blue) (held blue)"]
        solutions=["(grasp pr2R blue)",
                    "(grasp pr2L blue)"]
        numLoops=1*2*1
    
    elif nset==12:
        goalString=["(on blue table1) (on blue table1)"]
        if getEnvInfo(nenv, "blue") in [0,2]:
            solutions=["(grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1)"]
            numLoops=1*2*2
        elif getEnvInfo(nenv,"blue") in [0,1]:
            solutions=[]
            numLoops=0

    elif nset==13:
        goalString=["(on blue table2) (on blue table2)"]
        if getEnvInfo(nenv, "blue") in [0,1]:
            solutions=["(grasp pr2R blue) (place pr2R blue table2)",
                        "(grasp pr2L blue) (place pr2L blue table2)"]
            numLoops=1*2*2
        elif getEnvInfo(nenv,"blue") in [0,2]:
            solutions=[]
            numLoops=0

    elif nset==14:
        goalString=["(on blue red) (on blue red)"]
        solutions=["(grasp pr2R blue) (place pr2R blue red)",
                    "(grasp pr2L blue) (place pr2L blue red)"]
        numLoops=1*2*2

    elif nset==15:
        goalString=["(on blue green) (on blue green)"]
        solutions=["(grasp pr2R blue) (place pr2R blue green)",
                    "(grasp pr2L blue) (place pr2L blue green)"]
        numLoops=1*2*2

    return solutions, goalString, numLoops

def getData(nset=1, nenv=1):
    solutions=[]
    numLoops=0
    
    if nset==1:
        goalString=["(held red) (held green)", "(held green) (held red)"]
        solutions=["(grasp pr2R red) (grasp pr2L green)",
                    "(grasp pr2R green) (grasp pr2L red)",
                    "(grasp pr2L red) (grasp pr2R green)",
                    "(grasp pr2R green) (grasp pr2L red)"]
        numLoops=2*4*2

    elif nset==2:
        goalString=["(held red) (on green table1)", "(on green table1) (held red)"]
        if getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red)"]
            numLoops=2*8*3
        elif getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2R red)","(grasp pr2L red)"]
            numLoops=2*2*1

    elif nset==3:
        goalString=["(held red) (on green table2)", "(on green table2) (held red)"]
        if getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2L green table2)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green table2)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2R red)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2L red)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table2)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table2)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2R red)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2L red)"]
            numLoops=2*8*3
        elif getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2R red)","(grasp pr2L red)"]
            numLoops=2*2*1

    elif nset==4:
        goalString=["(held red) (on green red)", "(on green red) (held red)"]
        solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2L green red)",
                    "(grasp pr2R green) (grasp pr2L red) (place pr2R green red)",
                    "(grasp pr2R green) (place pr2R green red) (grasp pr2R red)",
                    "(grasp pr2L red) (grasp pr2R green) (place pr2R green red)",
                    "(grasp pr2L green) (grasp pr2R red) (place pr2L green red)",
                    "(grasp pr2L green) (place pr2L green red) (grasp pr2R red)"]
        numLoops=2*6*3

    elif nset==5:
        goalString=["(held red) (on green blue)", "(on green blue) (held red)"]
        solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2R green blue)",
                    "(grasp pr2R green) (place pr2R green blue) (grasp pr2R red)",
                    "(grasp pr2R green) (place pr2R green blue) (grasp pr2L red)",
                    "(grasp pr2L red) (grasp pr2R green) (place pr2R green blue)",
                    "(grasp pr2L green) (grasp pr2R red) (place pr2L green blue)",
                    "(grasp pr2L green) (place pr2L green blue) (grasp pr2R red)",
                    "(grasp pr2L green) (place pr2L green blue) (grasp pr2L red)",
                    "(grasp pr2R red) (grasp pr2L green) (place pr2L green blue)"]
        numLoops=2*8*3

    elif nset==6:
        goalString=["(held red) (held blue)", "(held blue) (held red)"]
        solutions=["(grasp pr2R red) (grasp pr2L blue)",
                    "(grasp pr2R blue) (grasp pr2L red)",
                    "(grasp pr2L red) (grasp pr2R blue)",
                    "(grasp pr2R blue) (grasp pr2L red)"]
        numLoops=2*4*2

    elif nset==7:
        goalString=["(held red) (on blue table1)", "(on blue table1) (held red)"]
        if getEnvInfo(nenv,"blue") in [0,2]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red)"]
            numLoops=2*8*3
        elif getEnvInfo(nenv,"blue") in [0,1]:
            solutions=["(grasp pr2R red)","(grasp pr2L red)"]
            numLoops=2*2*1

    elif nset==8:
        goalString=["(held red) (on blue table2)", "(on blue table2) (held red)"]
        if getEnvInfo(nenv,"blue") in [0,1]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table2)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table2)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2R red)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2L red)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table2)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table2)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2R red)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2L red)"]
            numLoops=2*8*3
        elif getEnvInfo(nenv,"blue") in [0,2]:
            solutions=["(grasp pr2R red)","(grasp pr2L red)"]
            numLoops=2*2*1

    elif nset==9:
        goalString=["(held red) (on blue red)", "(on blue red) (held red)"]
        solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2L blue red)",
                    "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue red)",
                    "(grasp pr2R blue) (place pr2R blue red) (grasp pr2R red)",
                    "(grasp pr2R blue) (place pr2R blue red) (grasp pr2L red)",
                    "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue red)",
                    "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue red)",
                    "(grasp pr2L blue) (place pr2L blue red) (grasp pr2R red)",
                    "(grasp pr2L blue) (place pr2L blue red) (grasp pr2L red)"]
        numLoops=2*8*3

    elif nset==10:
        goalString=["(held red) (on blue green)", "(on blue green) (held red)"]
        solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2L blue green)",
                    "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue green)",
                    "(grasp pr2R blue) (place pr2R blue green) (grasp pr2R red)",
                    "(grasp pr2R blue) (place pr2R blue green) (grasp pr2L red)",
                    "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue green)",
                    "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue green)",
                    "(grasp pr2L blue) (place pr2L blue green) (grasp pr2R red)",
                    "(grasp pr2L blue) (place pr2L blue green) (grasp pr2L red)"]
        numLoops=2*8*3

    elif nset==11:
        goalString=["(on red table1) (held green)", "(held green) (on red table1)"]
        if getEnvInfo(nenv,"red") in [0,2]:
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green)"]
            numLoops=2*8*3
        elif getEnvInfo(nenv,"red") in [0,1]:
            solutions=["(grasp pr2R green)","(grasp pr2L green)"]
            numLoops=2*2*1

    elif nset==12:
        goalString=["(on red table1) (on green table1)", "(on green table1) (on red table1)"]
        if getEnvInfo(nenv,"red") in [0,2] and getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (place pr2L red table1)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red table1) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red table1) (place pr2L green table1)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (place pr2R red table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table1) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table1) (place pr2R green table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"red") in [0,2] and getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2L red) (place pr2L red table1)",
                    "(grasp pr2R red) (place pr2R red table1)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"red") in [0,1] and getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2L green) (place pr2L green table1)",
                    "(grasp pr2R green) (place pr2R green table1)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"red") in [0,1] and getEnvInfo(nenv,"green") in [0,1]:
            solutions=[]
            numLoops=0

    elif nset==13:
        goalString=["(on red table1) (on green table2)", "(on green table2) (on red table1)"]
        if getEnvInfo(nenv,"red") in [0,2] and getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2L red table1) (place pr2R green table2)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green table2) (place pr2L red table1)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red table1) (place pr2L green table2)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table2) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green table2)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green table2)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table2) (place pr2R red table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table1) (place pr2L green table2)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table1) (place pr2R green table2)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table2) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green table2)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green table2)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"red") in [0,1] and getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2L green) (place pr2L green table2)",
                        "(grasp pr2R green) (place pr2R green table2)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"red") in [0,2] and getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R red) (place pr2R red table1)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"red") in [0,1] and getEnvInfo(nenv,"green") in [0,2]:
            solutions=[]
            numLoops=0

    elif nset==14:
        goalString=["(on red table1) (on green red)", "(on green red) (on red table1)"]
        if getEnvInfo(nenv,"red") in [0,2]:
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red table1) (place pr2L green red)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green red) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green red)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green red)", 
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green red) (place pr2L red table1)", 
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red table1) (place pr2R green red)", 
                        "(grasp pr2R green) (place pr2R green red) (grasp pr2R red) (place pr2R red table1)", 
                        "(grasp pr2R green) (place pr2R green red) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table1) (place pr2R green red)", 
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green red) (place pr2L red table1)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green red)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green red)", 
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green red) (place pr2R red table1)", 
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table1) (place pr2L green red)", 
                        "(grasp pr2L green) (place pr2L green red) (grasp pr2R red) (place pr2R red table1)", 
                        "(grasp pr2L green) (place pr2L green red) (grasp pr2L red) (place pr2L red table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"red") in [0,1]:
            solutions=["(grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2R green) (place pr2R green red)"]
            numLoops=2*2*2

    elif nset==15:
        goalString=["(on red table1) (on green blue)", "(on green blue) (on red table1)"]
        if getEnvInfo(nenv,"red") in [0,2]:
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red table1) (place pr2L green blue)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green blue) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green blue) (place pr2L red table1)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red table1) (place pr2R green blue)",
                        "(grasp pr2R green) (place pr2R green blue) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2R green) (place pr2R green blue) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table1) (place pr2R green blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green blue) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green blue) (place pr2R red table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table1) (place pr2L green blue)",
                        "(grasp pr2L green) (place pr2L green blue) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L green) (place pr2L green blue) (grasp pr2R red) (place pr2R red table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"red") in [0,1]:
            solutions=["(grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2R green) (place pr2R green blue)"]
            numLoops=2*2*2

    elif nset==16:
        goalString=["(on red table1) (held blue)", "(held blue) (on red table1)"]
        if getEnvInfo(nenv,"red") in [0,2]:
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue)"]
            numLoops=2*8*3
        elif getEnvInfo(nenv,"red") in [0,1]:
            solutions=["(grasp pr2R blue)","(grasp pr2L blue)"]
            numLoops=2*2*1

    elif nset==17:
        goalString=["(on red table1) (on blue table1)", "(on blue table1) (on red table1)"]
        if getEnvInfo(nenv,"red") in [0,2] and getEnvInfo(nenv,"blue") in [0,2]:
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (place pr2L red table1)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red table1) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red table1) (place pr2L blue table1)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (place pr2R red table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table1) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table1) (place pr2R blue table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"red") in [0,2] and getEnvInfo(nenv,"blue") in [0,1]:
            solutions=["(grasp pr2L red) (place pr2L red table1)",
                    "(grasp pr2R red) (place pr2R red table1)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"red") in [0,1] and getEnvInfo(nenv,"blue") in [0,2]:
            solutions=["(grasp pr2L blue) (place pr2L blue table1)",
                    "(grasp pr2R blue) (place pr2R blue table1)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"red") in [0,1] and getEnvInfo(nenv,"blue") in [0,1]:
            solutions=[]
            numLoops=0

    elif nset==18:
        goalString=["(on red table1) (on blue table2)", "(on blue table2) (on red table1)"]
        if getEnvInfo(nenv,"red") in [0,2] and getEnvInfo(nenv,"blue") in [0,1]:
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2L red table1) (place pr2R blue table2)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table2) (place pr2L red table1)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red table1) (place pr2L blue table2)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table2) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue table2)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue table2)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table2) (place pr2R red table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table1) (place pr2L blue table2)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table1) (place pr2R blue table2)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table2) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue table2)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue table2)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"red") in [0,1] and getEnvInfo(nenv,"blue") in [0,1]:
            solutions=["(grasp pr2L blue) (place pr2L blue table2)",
                        "(grasp pr2R blue) (place pr2R blue table2)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"red") in [0,2] and getEnvInfo(nenv,"blue") in [0,2]:
            solutions=["(grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R red) (place pr2R red table1)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"red") in [0,1] and getEnvInfo(nenv,"blue") in [0,2]:
            solutions=[]
            numLoops=0

    elif nset==19:
        goalString=["(on red table1) (on blue red)", "(on blue red) (on red table1)"]
        if getEnvInfo(nenv,"red") in [0,2]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red table1) (place pr2L blue red)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue red) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue red) (place pr2L red table1)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red table1) (place pr2R blue red)",
                        "(grasp pr2R blue) (place pr2R blue red) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2R blue) (place pr2R blue red) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table1) (place pr2R blue red)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue red) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue red) (place pr2R red table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table1) (place pr2L blue red)",
                        "(grasp pr2L blue) (place pr2L blue red) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L blue) (place pr2L blue red) (grasp pr2L red) (place pr2L red table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"red") in [0,1]:
            solutions=["(grasp pr2R blue) (place pr2R blue red)",
                    "(grasp pr2L blue) (place pr2L blue red)"]
            numLoops=2*2*2

    elif nset==20:
        goalString=["(on red table1) (on blue green)", "(on blue green) (on red table1)"]
        if getEnvInfo(nenv,"red") in [0,2]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red table1) (place pr2L blue green)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue green) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue green) (place pr2L red table1)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red table1) (place pr2R blue green)",
                        "(grasp pr2R blue) (place pr2R blue green) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R blue) (place pr2R blue green) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table1) (place pr2R blue green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue green) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue green) (place pr2R red table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table1) (place pr2L blue green)",
                        "(grasp pr2L blue) (place pr2L blue green) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L blue) (place pr2L blue green) (grasp pr2L red) (place pr2L red table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"red") in [0,1]:
            solutions=["(grasp pr2R blue) (place pr2R blue green)",
                    "(grasp pr2L blue) (place pr2L blue green)"]
            numLoops=2*2*2

    elif nset==21:
        goalString=["(on red table2) (held green)", "(held green) (on red table2)"]
        if getEnvInfo(nenv,"red") in [0,1]:
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2L red table2)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red table2)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2R green)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2L green)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table2)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table2)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2R green)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2L green)"]
            numLoops=2*8*3
        elif getEnvInfo(nenv,"red") in [0,2]:
            solutions=["(grasp pr2R green)","(grasp pr2L green)"]
            numLoops=2*2*1

    elif nset==22:
        goalString=["(on red table2) (on green table1)", "(on green table1) (on red table2)"]
        if getEnvInfo(nenv,"red") in [0,1] and getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (place pr2L red table2)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red table2) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red table2)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red table2)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red table2) (place pr2L green table1)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red table2)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (place pr2R red table2)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table2) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red table2)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red table2)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table2) (place pr2R green table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (place pr2L red table2)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2L green) (place pr2L green table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"red") in [0,1] and getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2L red) (place pr2L red table2)",
                    "(grasp pr2R red) (place pr2R red table2)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"red") in [0,2] and getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2L green) (place pr2L green table1)",
                    "(grasp pr2R green) (place pr2R green table1)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"red") in [0,2] and getEnvInfo(nenv,"green") in [0,1]:
            solutions=[]
            numLoops=0

    elif nset==23:
        goalString=["(on red table2) (on green table2)", "(on green table2) (on red table2)"]
        if getEnvInfo(nenv,"red") in [0,1] and getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2R green table2) (place pr2L red table2)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red table2) (place pr2R green table2)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2R red) (place pr2R red table2)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2L red) (place pr2L red table2)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red table2) (place pr2L green table2)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table2) (place pr2R red table2)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2R green) (place pr2R green table2)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2L green) (place pr2L green table2)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table2) (place pr2R red table2)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table2) (place pr2L green table2)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2R red) (place pr2R red table2)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2L red) (place pr2L red table2)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table2) (place pr2R green table2)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table2) (place pr2L red table2)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2R green) (place pr2R green table2)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2L green) (place pr2L green table2)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"red") in [0,1] and getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2L red) (place pr2L red table2)",
                    "(grasp pr2R red) (place pr2R red table2)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"red") in [0,2] and getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2L green) (place pr2L green table2)",
                    "(grasp pr2R green) (place pr2R green table2)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"red") in [0,2] and getEnvInfo(nenv,"green") in [0,2]:
            solutions=[]
            numLoops=0

    elif nset==24:
        goalString=["(on red table2) (on green red)", "(on green red) (on red table2)"]
        if getEnvInfo(nenv,"red") in [0,1]:
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red table2) (place pr2L green red)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green red) (place pr2R red table2)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green red) (place pr2L red table2)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red table2) (place pr2R green red)",
                        "(grasp pr2R green) (place pr2R green red) (grasp pr2R red) (place pr2R red table2)",
                        "(grasp pr2R green) (place pr2R green red) (grasp pr2L red) (place pr2L red table2)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table2) (place pr2R green red)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green red) (place pr2L red table2)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green red) (place pr2R red table2)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table2) (place pr2L green red)",
                        "(grasp pr2L green) (place pr2L green red) (grasp pr2L red) (place pr2L red table2)",
                        "(grasp pr2L green) (place pr2L green red) (grasp pr2R red) (place pr2R red table2)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"red") in [0,2]:
            solutions=["(grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2R green) (place pr2R green red)"]
            numLoops=2*2*2

    elif nset==25:
        goalString=["(on red table2) (on green blue)", "(on green blue) (on red table2)"]
        if getEnvInfo(nenv,"red") in [0,1]:
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red table2) (place pr2L green blue)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green blue) (place pr2R red table2)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green blue) (place pr2L red table2)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red table2) (place pr2R green blue)",
                        "(grasp pr2R green) (place pr2R green blue) (grasp pr2R red) (place pr2R red table2)",
                        "(grasp pr2R green) (place pr2R green blue) (grasp pr2L red) (place pr2L red table2)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table2) (place pr2R green blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green blue) (place pr2L red table2)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table2) (place pr2L green blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green blue) (place pr2R red table2)",
                        "(grasp pr2L green) (place pr2L green blue) (grasp pr2R red) (place pr2R red table2)",
                        "(grasp pr2L green) (place pr2L green blue) (grasp pr2L red) (place pr2L red table2)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"red") in [0,2]:
            solutions=["(grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2R green) (place pr2R green blue)"]
            numLoops=2*2*2

    elif nset==26:
        goalString=["(on red table2) (held blue)", "(held blue) (on red table2)"]
        if getEnvInfo(nenv,"red") in [0,1]:
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2L red table2)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red table2)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2R blue)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2L blue)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table2)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table2)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2R blue)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2L blue)"]
            numLoops=2*8*3
        elif getEnvInfo(nenv,"red") in [0,2]:
            solutions=["(grasp pr2R blue)","(grasp pr2L blue)"]
            numLoops=2*2*1

    elif nset==27:
        goalString=["(on red table2) (on blue table1)", "(on blue table1) (on red table2)"]
        if getEnvInfo(nenv,"red") in [0,1] and getEnvInfo(nenv,"blue") in [0,2]:
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (place pr2L red table2)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red table2) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red table2)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red table2)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red table2) (place pr2L blue table1)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (place pr2R red table2)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (place pr2R red table2)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table2) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red table2)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red table2)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table2) (place pr2R blue table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (place pr2L red table2)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2L blue) (place pr2L blue table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"red") in [0,1] and getEnvInfo(nenv,"blue") in [0,1]:
            solutions=["(grasp pr2L red) (place pr2L red table2)",
                    "(grasp pr2R red) (place pr2R red table2)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"red") in [0,2] and getEnvInfo(nenv,"blue") in [0,2]:
            solutions=["(grasp pr2L blue) (place pr2L blue table1)",
                    "(grasp pr2R blue) (place pr2R blue table1)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"red") in [0,2] and getEnvInfo(nenv,"blue") in [0,1]:
            solutions=[]
            numLoops=0

    elif nset==28:
        goalString=["(on red table2) (on blue table2)", "(on blue table2) (on red table2)"]
        if getEnvInfo(nenv,"red") in [0,1] and getEnvInfo(nenv,"blue") in [0,1]:
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table2) (place pr2L red table2)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red table2) (place pr2R blue table2)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2R red) (place pr2R red table2)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2L red) (place pr2L red table2)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red table2) (place pr2L blue table2)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table2) (place pr2R red table2)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2R blue) (place pr2R blue table2)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2L blue) (place pr2L blue table2)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table2) (place pr2R red table2)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table2) (place pr2L blue table2)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2R red) (place pr2R red table2)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2L red) (place pr2L red table2)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table2) (place pr2R blue table2)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table2) (place pr2L red table2)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2R blue) (place pr2R blue table2)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2L blue) (place pr2L blue table2)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"red") in [0,1] and getEnvInfo(nenv,"blue") in [0,2]:
            solutions=["(grasp pr2L red) (place pr2L red table2)",
                    "(grasp pr2R red) (place pr2R red table2)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"red") in [0,2] and getEnvInfo(nenv,"blue") in [0,1]:
            solutions=["(grasp pr2L blue) (place pr2L blue table2)",
                    "(grasp pr2R blue) (place pr2R blue table2)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"red") in [0,2] and getEnvInfo(nenv,"blue") in [0,2]:
            solutions=[]
            numLoops=0

    elif nset==29:
        goalString=["(on red table2) (on blue red)", "(on blue red) (on red table2)"]
        if getEnvInfo(nenv,"red") in [0,1]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red table2) (place pr2L blue red)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue red) (place pr2R red table2)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue red) (place pr2L red table2)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red table2) (place pr2R blue red)",
                        "(grasp pr2R blue) (place pr2R blue red) (grasp pr2R red) (place pr2R red table2)",
                        "(grasp pr2R blue) (place pr2R blue red) (grasp pr2L red) (place pr2L red table2)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table2) (place pr2R blue red)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue red) (place pr2L red table2)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue red) (place pr2R red table2)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table2) (place pr2L blue red)",
                        "(grasp pr2L blue) (place pr2L blue red) (grasp pr2L red) (place pr2L red table2)",
                        "(grasp pr2L blue) (place pr2L blue red) (grasp pr2R red) (place pr2R red table2)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"red") in [0,2]:
            solutions=["(grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2R blue) (place pr2R blue red)"]
            numLoops=2*2*2

    elif nset==30:
        goalString=["(on red table2) (on blue green)", "(on blue green) (on red table2)"]
        if getEnvInfo(nenv,"red") in [0,1]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red table2) (place pr2L blue green)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue green) (place pr2R red table2)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R red) (place pr2R red table2) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue green) (place pr2L red table2)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red table2) (place pr2R blue green)",
                        "(grasp pr2R blue) (place pr2R blue green) (grasp pr2L red) (place pr2L red table2)",
                        "(grasp pr2R blue) (place pr2R blue green) (grasp pr2R red) (place pr2R red table2)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table2) (place pr2R blue green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue green) (place pr2L red table2)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L red) (place pr2L red table2) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue green) (place pr2R red table2)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table2) (place pr2L blue green)",
                        "(grasp pr2L blue) (place pr2L blue green) (grasp pr2R red) (place pr2R red table2)",
                        "(grasp pr2L blue) (place pr2L blue green) (grasp pr2L red) (place pr2L red table2)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"red") in [0,2]:
            solutions=["(grasp pr2R blue) (place pr2R blue green)",
                    "(grasp pr2L blue) (place pr2L blue green)"]
            numLoops=2*2*2

    elif nset==31:
        goalString=["(on red green) (held green)", "(held green) (on red green)"]
        solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red green)",
                    "(grasp pr2R red) (place pr2R red green) (grasp pr2R green)",
                    "(grasp pr2R red) (place pr2R red green) (grasp pr2L green)",
                    "(grasp pr2R green) (grasp pr2L red) (place pr2L red green)",
                    "(grasp pr2L red) (place pr2L red green) (grasp pr2R green)",
                    "(grasp pr2L red) (place pr2L red green) (grasp pr2L green)",
                    "(grasp pr2L red) (grasp pr2R green) (place pr2L red green)",
                    "(grasp pr2L green) (grasp pr2R red) (place pr2R red green)"]
        numLoops=2*8*3

    elif nset==32:
        goalString=["(on red green) (on green table1)", "(on green table1) (on red green)"]
        if getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green table1)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red green)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red green)",
                        "(grasp pr2R red) (place pr2R red green) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (place pr2L red green)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red green) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (place pr2L red green)",
                        "(grasp pr2L red) (place pr2L red green) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L red) (place pr2L red green) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (place pr2R red green)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red green) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red green)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L red) (place pr2L red green)"]
            numLoops=2*2*2


    elif nset==33:
        goalString=["(on red green) (on green table2)", "(on green table2) (on red green)"]
        if getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green table2)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table2) (place pr2R red green)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table2) (place pr2R red green)",
                        "(grasp pr2R red) (place pr2R red green) (grasp pr2R green) (place pr2R green table2)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green table2) (place pr2L red green)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red green) (place pr2R green table2)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green table2)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table2) (place pr2L red green)",
                        "(grasp pr2L red) (place pr2L red green) (grasp pr2R green) (place pr2R green table2)",
                        "(grasp pr2L red) (place pr2L red green) (grasp pr2L green) (place pr2L green table2)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table2) (place pr2R red green)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red green) (place pr2L green table2)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2L red) (place pr2L red green)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L red) (place pr2L red green)"]
            numLoops=2*2*2

    elif nset==34:
        goalString=["(on red green) (on green blue)", "(on green blue) (on red green)"]
        solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green blue)",
                    "(grasp pr2R red) (grasp pr2L green) (place pr2L green blue) (place pr2R red green)",
                    "(grasp pr2R red) (place pr2R red green) (grasp pr2R green) (place pr2R green blue)",
                    "(grasp pr2R red) (place pr2R red green) (grasp pr2L green) (place pr2L green blue)",
                    "(grasp pr2R green) (grasp pr2L red) (place pr2R green blue) (place pr2L red green)",
                    "(grasp pr2R green) (grasp pr2L red) (place pr2L red green) (place pr2R green blue)",
                    "(grasp pr2R green) (place pr2R green blue) (grasp pr2R red) (place pr2R red green)",
                    "(grasp pr2R green) (place pr2R green blue) (grasp pr2L red) (place pr2L red green)",
                    "(grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green blue)",
                    "(grasp pr2L red) (grasp pr2R green) (place pr2R green blue) (place pr2L red green)",
                    "(grasp pr2L red) (place pr2L red green) (grasp pr2R green) (place pr2R green blue)",
                    "(grasp pr2L red) (place pr2L red green) (grasp pr2L green) (place pr2L green blue)",
                    "(grasp pr2L green) (grasp pr2R red) (place pr2R red green) (place pr2L green blue)",
                    "(grasp pr2L green) (grasp pr2R red) (place pr2L green blue) (place pr2R red green)",
                    "(grasp pr2L green) (place pr2L green blue) (grasp pr2R red) (place pr2R red green)",
                    "(grasp pr2L green) (place pr2L green blue) (grasp pr2L red) (place pr2L red green)"]
        numLoops=2*16*4

    elif nset==35:
        goalString=["(on red green) (held blue)", "(held blue) (on red green)"]
        solutions=["(grasp pr2R red) (place pr2R red green) (grasp pr2R blue)",
                    "(grasp pr2R red) (place pr2R red green) (grasp pr2L blue)",
                    "(grasp pr2R blue) (grasp pr2L red) (place pr2L red green)",
                    "(grasp pr2L red) (grasp pr2R blue) (place pr2L red green)",
                    "(grasp pr2L red) (place pr2L red green) (grasp pr2R blue)",
                    "(grasp pr2L red) (place pr2L red green) (grasp pr2L blue)",
                    "(grasp pr2L blue) (grasp pr2R red) (place pr2R red green)",
                    "(grasp pr2R blue) (grasp pr2L red) (place pr2L red green)"]
        numLoops=2*8*3

    elif nset==36:
        goalString=["(on red green) (on blue table1)", "(on blue table1) (on red green)"]
        if getEnvInfo(nenv,"blue") in [0,2]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red green) (place pr2L blue table1)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (place pr2R red green)",
                        "(grasp pr2R red) (place pr2R red green) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R red) (place pr2R red green) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (place pr2L red green)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red green) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red green) (place pr2R blue table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (place pr2L red green)",
                        "(grasp pr2L red) (place pr2L red green) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L red) (place pr2L red green) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (place pr2R red green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red green) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red green)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"blue") in [0,1]:
            solutions=["(grasp pr2R red) (place pr2R red green)",
                    "(grasp pr2L red) (place pr2L red green)"]
            numLoops=2*2*2

    elif nset==37:
        goalString=["(on red green) (on blue table2)", "(on blue table2) (on red green)"]
        if getEnvInfo(nenv,"blue") in [0,1]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red green) (place pr2L blue table2)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table2) (place pr2R red green)",
                        "(grasp pr2R red) (place pr2R red green) (grasp pr2R blue) (place pr2R blue table2)",
                        "(grasp pr2R red) (place pr2R red green) (grasp pr2L blue) (place pr2L blue table2)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table2) (place pr2L red green)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red green) (place pr2R blue table2)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red green) (place pr2R blue table2)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table2) (place pr2L red green)",
                        "(grasp pr2L red) (place pr2L red green) (grasp pr2R blue) (place pr2R blue table2)",
                        "(grasp pr2L red) (place pr2L red green) (grasp pr2L blue) (place pr2L blue table2)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table2) (place pr2R red green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red green) (place pr2L blue table2)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2L red) (place pr2L red green)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"blue") in [0,2]:
            solutions=["(grasp pr2R red) (place pr2R red green)",
                    "(grasp pr2L red) (place pr2L red green)"]
            numLoops=2*2*2

    elif nset==38:
        goalString=["(on red green) (on blue red)", "(on blue red) (on red green)"]
        solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red green) (place pr2L blue red)",
                "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue red) (place pr2R red green)",
                "(grasp pr2R red) (place pr2R red green) (grasp pr2R blue) (place pr2R blue red)",
                "(grasp pr2R red) (place pr2R red green) (grasp pr2L blue) (place pr2L blue red)",
                "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue red) (place pr2L red green)",
                "(grasp pr2R blue) (grasp pr2L red) (place pr2L red green) (place pr2R blue red)",
                "(grasp pr2R blue) (place pr2R blue red) (grasp pr2R red) (place pr2R red green)",
                "(grasp pr2R blue) (place pr2R blue red) (grasp pr2L red) (place pr2L red green)",
                "(grasp pr2L red) (place pr2L red green) (grasp pr2L blue) (place pr2L blue red)",
                "(grasp pr2L red) (place pr2L red green) (grasp pr2R blue) (place pr2R blue red)",
                "(grasp pr2L red) (grasp pr2R blue) (place pr2L red green) (place pr2R blue red)",
                "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue red) (place pr2L red green)",
                "(grasp pr2L blue) (grasp pr2R red) (place pr2R red green) (place pr2L blue red)",
                "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue red) (place pr2R red green)",
                "(grasp pr2L blue) (place pr2L blue red) (grasp pr2R red) (place pr2R red green)",
                "(grasp pr2L blue) (place pr2L blue red) (grasp pr2L red) (place pr2L red green)"]
        numLoops=2*16*4

    elif nset==39:
        goalString=["(on red green) (on blue green)", "(on blue green) (on red green)"]
        solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red green) (place pr2L blue green)",
                    "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue green) (place pr2R red green)",
                    "(grasp pr2R red) (place pr2R red green) (grasp pr2R blue) (place pr2R blue green)",
                    "(grasp pr2R red) (place pr2R red green) (grasp pr2L blue) (place pr2L blue green)",
                    "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue green) (place pr2L red green)",
                    "(grasp pr2R blue) (grasp pr2L red) (place pr2L red green) (place pr2R blue green)",
                    "(grasp pr2R blue) (place pr2R blue green) (grasp pr2R red) (place pr2R red green)",
                    "(grasp pr2R blue) (place pr2R blue green) (grasp pr2L red) (place pr2L red green)",
                    "(grasp pr2L red) (grasp pr2R blue) (place pr2L red green) (place pr2R blue green)",
                    "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue green) (place pr2L red green)",
                    "(grasp pr2L red) (place pr2L red green) (grasp pr2R blue) (place pr2R blue green)",
                    "(grasp pr2L red) (place pr2L red green) (grasp pr2L blue) (place pr2L blue green)",
                    "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue green) (place pr2R red green)",
                    "(grasp pr2L blue) (grasp pr2R red) (place pr2R red green) (place pr2L blue green)",
                    "(grasp pr2L blue) (place pr2L blue green) (grasp pr2R red) (place pr2R red green)",
                    "(grasp pr2L blue) (place pr2L blue green) (grasp pr2L red) (place pr2L red green)"]
        numLoops=2*16*4

    elif nset==40:
        goalString=["(on red blue) (held green)", "(held green) (on red blue)"]
        solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red blue)",
                    "(grasp pr2R red) (place pr2R red blue) (grasp pr2L green)",
                    "(grasp pr2R red) (grasp pr2L green) (place pr2R red blue)",
                    "(grasp pr2R green) (grasp pr2L red) (place pr2L red blue)",
                    "(grasp pr2L red) (grasp pr2R green) (place pr2L red blue)",
                    "(grasp pr2L red) (place pr2L red blue) (grasp pr2R green)",
                    "(grasp pr2L red) (place pr2L red blue) (grasp pr2L green)",
                    "(grasp pr2L green) (grasp pr2R red) (place pr2R red blue)"]
        numLoops=2*8*3

    elif nset==41:
        goalString=["(on red blue) (on green table1)", "(on green table1) (on red blue)"]
        if getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red blue) (place pr2L green table1)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red blue)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red blue)",
                        "(grasp pr2R red) (place pr2R red blue) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (place pr2L red blue)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red blue) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red blue) (place pr2R green table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (place pr2L red blue)",
                        "(grasp pr2L red) (place pr2L red blue) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L red) (place pr2L red blue) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (place pr2R red blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red blue) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red blue)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L red) (place pr2L red blue)"]
            numLoops=2*2*2

    elif nset==42:
        goalString=["(on red blue) (on green table2)", "(on green table2) (on red blue)"]
        if getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red blue) (place pr2L green table2)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table2) (place pr2R red blue)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table2) (place pr2R red blue)",
                        "(grasp pr2R red) (place pr2R red blue) (grasp pr2R green) (place pr2R green table2)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green table2) (place pr2L red blue)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red blue) (place pr2R green table2)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red blue) (place pr2R green table2)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table2) (place pr2L red blue)",
                        "(grasp pr2L red) (place pr2L red blue) (grasp pr2R green) (place pr2R green table2)",
                        "(grasp pr2L red) (place pr2L red blue) (grasp pr2L green) (place pr2L green table2)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table2) (place pr2R red blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red blue) (place pr2L green table2)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2L red) (place pr2L red blue)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L red) (place pr2L red blue)"]
            numLoops=2*2*2

    elif nset==43:
        goalString=["(on red blue) (on green red)", "(on green red) (on red blue)"]
        solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red blue) (place pr2L green red)",
                    "(grasp pr2R red) (grasp pr2L green) (place pr2L green red) (place pr2R red blue)",
                    "(grasp pr2R red) (place pr2R red blue) (grasp pr2L green) (place pr2L green red)",
                    "(grasp pr2R red) (place pr2R red blue) (grasp pr2R green) (place pr2R green red)",
                    "(grasp pr2R green) (grasp pr2L red) (place pr2R green red) (place pr2L red blue)",
                    "(grasp pr2R green) (grasp pr2L red) (place pr2L red blue) (place pr2R green red)",
                    "(grasp pr2R green) (place pr2R green red) (grasp pr2R red) (place pr2R red blue)",
                    "(grasp pr2R green) (place pr2R green red) (grasp pr2L red) (place pr2L red blue)",
                    "(grasp pr2L red) (grasp pr2R green) (place pr2L red blue) (place pr2R green red)",
                    "(grasp pr2L red) (grasp pr2R green) (place pr2R green red) (place pr2L red blue)",
                    "(grasp pr2L red) (place pr2L red blue) (grasp pr2R green) (place pr2R green red)",
                    "(grasp pr2L red) (place pr2L red blue) (grasp pr2L green) (place pr2L green red)",
                    "(grasp pr2L green) (grasp pr2R red) (place pr2L green red) (place pr2R red blue)",
                    "(grasp pr2L green) (grasp pr2R red) (place pr2R red blue) (place pr2L green red)",
                    "(grasp pr2L green) (place pr2L green red) (grasp pr2R red) (place pr2R red blue)",
                    "(grasp pr2L green) (place pr2L green red) (grasp pr2L red) (place pr2L red blue)"]
        numLoops=2*16*4

    elif nset==44:
        goalString=["(on red blue) (on green blue)", "(on green blue) (on red blue)"]
        solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red blue) (place pr2L green blue)",
                    "(grasp pr2R red) (grasp pr2L green) (place pr2L green blue) (place pr2R red blue)",
                    "(grasp pr2R red) (place pr2R red blue) (grasp pr2L green) (place pr2L green blue)",
                    "(grasp pr2R red) (place pr2R red blue) (grasp pr2R green) (place pr2R green blue)",
                    "(grasp pr2R green) (grasp pr2L red) (place pr2R green blue) (place pr2L red blue)",
                    "(grasp pr2R green) (grasp pr2L red) (place pr2L red blue) (place pr2R green blue)",
                    "(grasp pr2R green) (place pr2R green blue) (grasp pr2R red) (place pr2R red blue)",
                    "(grasp pr2R green) (place pr2R green blue) (grasp pr2L red) (place pr2L red blue)",
                    "(grasp pr2L red) (grasp pr2R green) (place pr2L red blue) (place pr2R green blue)",
                    "(grasp pr2L red) (grasp pr2R green) (place pr2R green blue) (place pr2L red blue)",
                    "(grasp pr2L red) (place pr2L red blue) (grasp pr2R green) (place pr2R green blue)",
                    "(grasp pr2L red) (place pr2L red blue) (grasp pr2L green) (place pr2L green blue)",
                    "(grasp pr2L green) (grasp pr2R red) (place pr2L green blue) (place pr2R red blue)",
                    "(grasp pr2L green) (grasp pr2R red) (place pr2R red blue) (place pr2L green blue)",
                    "(grasp pr2L green) (place pr2L green blue) (grasp pr2R red) (place pr2R red blue)",
                    "(grasp pr2L green) (place pr2L green blue) (grasp pr2L red) (place pr2L red blue)"]
        numLoops=2*16*4

    elif nset==45:
        goalString=["(on red blue) (held blue)", "(held blue) (on red blue)"]
        solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red blue)",
                    "(grasp pr2R red) (place pr2R red blue) (grasp pr2R blue)",
                    "(grasp pr2R red) (place pr2R red blue) (grasp pr2L blue)",
                    "(grasp pr2R blue) (grasp pr2L red) (place pr2L red blue)",
                    "(grasp pr2L red) (grasp pr2R blue) (place pr2L red blue)",
                    "(grasp pr2L red) (place pr2L red blue) (grasp pr2R blue)",
                    "(grasp pr2L red) (place pr2L red blue) (grasp pr2L blue)",
                    "(grasp pr2L blue) (grasp pr2R red) (place pr2R red blue)"]
        numLoops=2*8*3

    elif nset==46:
        goalString=["(on red blue) (on blue table1)", "(on blue table1) (on red blue)"]
        if getEnvInfo(nenv,"blue") in [0,2]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue table1)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (place pr2R red blue)",
                        "(grasp pr2R red) (place pr2R red blue) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R red) (place pr2R red blue) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (place pr2L red blue)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red blue) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (place pr2L red blue)",
                        "(grasp pr2L red) (place pr2L red blue) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L red) (place pr2L red blue) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (place pr2R red blue)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red blue) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red blue)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"blue") in [0,1]:
            solutions=["(grasp pr2R red) (place pr2R red blue)",
                    "(grasp pr2L red) (place pr2L red blue)"]
            numLoops=2*2*2

    elif nset==47:
        goalString=["(on red blue) (on blue table2)", "(on blue table2) (on red blue)"]
        if getEnvInfo(nenv,"blue") in [0,1]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue table2)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table2) (place pr2R red blue)",
                        "(grasp pr2R red) (place pr2R red blue) (grasp pr2R blue) (place pr2R blue table2)",
                        "(grasp pr2R red) (place pr2R red blue) (grasp pr2L blue) (place pr2L blue table2)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table2) (place pr2L red blue)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red blue) (place pr2R blue table2)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue table2)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table2) (place pr2L red blue)",
                        "(grasp pr2L red) (place pr2L red blue) (grasp pr2R blue) (place pr2R blue table2)",
                        "(grasp pr2L red) (place pr2L red blue) (grasp pr2L blue) (place pr2L blue table2)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table2) (place pr2R red blue)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red blue) (place pr2L blue table2)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2L red) (place pr2L red blue)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"blue") in [0,2]:
            solutions=["(grasp pr2R red) (place pr2R red blue)",
                    "(grasp pr2L red) (place pr2L red blue)"]
            numLoops=2*2*2

    elif nset==48:
        goalString=["(on red blue) (on blue green)", "(on blue green) (on red blue)"]
        solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue green)",
                    "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue green) (place pr2R red blue)",
                    "(grasp pr2R red) (place pr2R red blue) (grasp pr2R blue) (place pr2R blue green)",
                    "(grasp pr2R red) (place pr2R red blue) (grasp pr2L blue) (place pr2L blue green)",
                    "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue green) (place pr2L red blue)",
                    "(grasp pr2R blue) (grasp pr2L red) (place pr2L red blue) (place pr2R blue green)",
                    "(grasp pr2R blue) (place pr2R blue green) (grasp pr2R red) (place pr2R red blue)",
                    "(grasp pr2R blue) (place pr2R blue green) (grasp pr2L red) (place pr2L red blue)",
                    "(grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue green)",
                    "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue green) (place pr2L red blue)",
                    "(grasp pr2L red) (place pr2L red blue) (grasp pr2R blue) (place pr2R blue green)",
                    "(grasp pr2L red) (place pr2L red blue) (grasp pr2L blue) (place pr2L blue green)",
                    "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue green) (place pr2R red blue)",
                    "(grasp pr2L blue) (grasp pr2R red) (place pr2R red blue) (place pr2L blue green)",
                    "(grasp pr2L blue) (place pr2L blue green) (grasp pr2R red) (place pr2R red blue)",
                    "(grasp pr2L blue) (place pr2L blue green) (grasp pr2L red) (place pr2L red blue)"]
        numLoops=2*16*4

    elif nset==49:
        goalString=["(held green) (held blue)", "(held blue) (held green)"]
        solutions=["(grasp pr2R blue) (grasp pr2L green)",
                    "(grasp pr2L blue) (grasp pr2R green)",
                    "(grasp pr2R green) (grasp pr2L blue)",
                    "(grasp pr2L green) (grasp pr2R blue)"]
        numLoops=2*4*2

    elif nset==50:
        goalString=["(held green) (on blue table1)", "(on blue table1) (held green)"]
        if getEnvInfo(nenv,"blue") in [0,2]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green)"]
            numLoops=2*8*3
        elif getEnvInfo(nenv,"blue") in [0,1]:
            solutions=["(grasp pr2R green)","(grasp pr2L green)"]
            numLoops=2*2*1

    elif nset==51:
        goalString=["(held green) (on blue table2)", "(on blue table2) (held green)"]
        if getEnvInfo(nenv,"blue") in [0,1]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table2)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table2)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2R green)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2L green)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table2)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table2)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2R green)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2L green)"]
            numLoops=2*8*3
        elif getEnvInfo(nenv,"blue") in [0,2]:
            solutions=["(grasp pr2R green)","(grasp pr2L green)"]
            numLoops=2*2*1

    elif nset==52:
        goalString=["(held green) (on blue red)", "(on blue red) (held green)"]
        solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2L blue red)",
                    "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue red)",
                    "(grasp pr2R blue) (place pr2R blue red) (grasp pr2R green)",
                    "(grasp pr2R blue) (place pr2R blue red) (grasp pr2L green)",
                    "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue red)",
                    "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue red)",
                    "(grasp pr2L blue) (place pr2L blue red) (grasp pr2R green)",
                    "(grasp pr2L blue) (place pr2L blue red) (grasp pr2L green)"]
        numLoops=2*8*3

    elif nset==53:
        goalString=["(held green) (on blue green)", "(on blue green) (held green)"]
        solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2L blue green)",
                    "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue green)",
                    "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue green)",
                    "(grasp pr2R blue) (place pr2R blue green) (grasp pr2R green)",
                    "(grasp pr2R blue) (place pr2R blue green) (grasp pr2L green)",
                    "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue green)",
                    "(grasp pr2L blue) (place pr2L blue green) (grasp pr2R green)",
                    "(grasp pr2L blue) (place pr2L blue green) (grasp pr2L green)"]
        numLoops=2*8*3

    elif nset==54:
        goalString=["(on green table1) (held blue)", "(held blue) (on green table1)"]
        if getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2R blue) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue)"]
            numLoops=2*8*3
        elif getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2R blue)","(grasp pr2L blue)"]
            numLoops=2*2*1

    elif nset==55:
        goalString=["(on green table1) (on blue table1)", "(on blue table1) (on green table1)"]
        if getEnvInfo(nenv,"blue") in [0,2] and getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green table1) (place pr2L blue table1)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (place pr2L green table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green table1) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table1) (place pr2R blue table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (place pr2R green table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table1) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"blue") in [0,2] and getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2L blue) (place pr2L blue table1)",
                    "(grasp pr2R blue) (place pr2R blue table1)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"blue") in [0,1] and getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2L green) (place pr2L green table1)",
                    "(grasp pr2R green) (place pr2R green table1)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"blue") in [0,1] and getEnvInfo(nenv,"green") in [0,1]:
            solutions=[]
            numLoops=0        

    elif nset==56:
        goalString=["(on green table1) (on blue table2)", "(on blue table2) (on green table1)"]
        if getEnvInfo(nenv,"blue") in [0,1] and getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green table1) (place pr2L blue table2)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table2) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue table2)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue table2)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table2) (place pr2L green table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green table1) (place pr2R blue table2)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table1) (place pr2R blue table2)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table2) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue table2)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue table2)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table2) (place pr2R green table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table1) (place pr2L blue table2)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2L green) (place pr2L green table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"blue") in [0,1] and getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2L blue) (place pr2L blue table2)",
                    "(grasp pr2R blue) (place pr2R blue table2)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"blue") in [0,2] and getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2L green) (place pr2L green table1)",
                    "(grasp pr2R green) (place pr2R green table1)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"blue") in [0,2] and getEnvInfo(nenv,"green") in [0,1]:
            solutions=[]
            numLoops=0

    elif nset==57:
        goalString=["(on green table1) (on blue red)", "(on blue red) (on green table1)"]
        if getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green table1) (place pr2L blue red)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue red) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue red) (place pr2L green table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green table1) (place pr2R blue red)",
                        "(grasp pr2R blue) (place pr2R blue red) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2R blue) (place pr2R blue red) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table1) (place pr2R blue red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue red) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue red) (place pr2R green table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table1) (place pr2L blue red)",
                        "(grasp pr2L blue) (place pr2L blue red) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L blue) (place pr2L blue red) (grasp pr2L green) (place pr2L green table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2R blue) (place pr2R blue red)"]
            numLoops=2*2*2

    elif nset==58:
        goalString=["(on green table1) (on blue green)", "(on blue green) (on green table1)"]
        if getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green table1) (place pr2L blue green)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue green) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue green) (place pr2L green table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green table1) (place pr2R blue green)",
                        "(grasp pr2R blue) (place pr2R blue green) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2R blue) (place pr2R blue green) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table1) (place pr2R blue green)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue green) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue green) (place pr2R green table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table1) (place pr2L blue green)",
                        "(grasp pr2L blue) (place pr2L blue green) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L blue) (place pr2L blue green) (grasp pr2L green) (place pr2L green table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2R blue) (place pr2R blue green)"]
            numLoops=2*2*2

    elif nset==59:
        goalString=["(on green table2) (held blue)", "(held blue) (on green table2)"]
        if getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green table2)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2R blue)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2L blue)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green table2)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table2)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2R blue)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2L blue)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table2)"]
            numLoops=2*8*3
        elif getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2L blue)",
                    "(grasp pr2R blue)"]
            numLoops=2*2*1

    elif nset==60:
        goalString=["(on green table2) (on blue table1)", "(on blue table1) (on green table2)"]
        if getEnvInfo(nenv,"blue") in [0,2] and getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (place pr2R green table2)",
                        "(grasp pr2R green) (grasp pr2L blue)  (place pr2R green table2)(place pr2L blue table1)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (place pr2L green table2)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green table2) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green table2)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green table2)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table2) (place pr2R blue table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (place pr2L green table2)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (place pr2R green table2)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table2) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green table2)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green table2)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"blue") in [0,1] and getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2L green) (place pr2L green table2)",
                        "(grasp pr2R green) (place pr2R green table2)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"blue") in [0,2] and getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"blue") in [0,1] and getEnvInfo(nenv,"green") in [0,2]:
            solutions=[]
            numLoops=0

    elif nset==61:
        goalString=["(on green table2) (on blue table2)", "(on blue table2) (on green table2)"]
        if getEnvInfo(nenv,"blue") in [0,1] and getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green table2) (place pr2L blue table2)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table2) (place pr2R green table2)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2R blue) (place pr2R blue table2)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2L blue) (place pr2L blue table2)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table2) (place pr2L green table2)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green table2) (place pr2R blue table2)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2R green) (place pr2R green table2)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2L green) (place pr2L green table2)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table2) (place pr2R blue table2)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table2) (place pr2L green table2)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2R blue) (place pr2R blue table2)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2L blue) (place pr2L blue table2)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table2) (place pr2R green table2)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table2) (place pr2L blue table2)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2R green) (place pr2R green table2)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2L green) (place pr2L green table2)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"blue") in [0,1] and getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2L blue) (place pr2L blue table2)",
                    "(grasp pr2R blue) (place pr2R blue table2)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"blue") in [0,2] and getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2L green) (place pr2L green table2)",
                    "(grasp pr2R green) (place pr2R green table2)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"blue") in [0,2] and getEnvInfo(nenv,"green") in [0,2]:
            solutions=[]
            numLoops=0

    elif nset==62:
        goalString=["(on green table2) (on blue red)", "(on blue red) (on green table2)"]
        if getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green table2) (place pr2L blue red)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue red) (place pr2R green table2)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue red) (place pr2L green table2)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green table2) (place pr2R blue red)",
                        "(grasp pr2R blue) (place pr2R blue red) (grasp pr2R green) (place pr2R green table2)",
                        "(grasp pr2R blue) (place pr2R blue red) (grasp pr2L green) (place pr2L green table2)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table2) (place pr2R blue red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue red) (place pr2L green table2)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue red) (place pr2R green table2)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table2) (place pr2L blue red)",
                        "(grasp pr2L blue) (place pr2L blue red) (grasp pr2L green) (place pr2L green table2)",
                        "(grasp pr2L blue) (place pr2L blue red) (grasp pr2R green) (place pr2R green table2)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2R blue) (place pr2R blue red)"]
            numLoops=2*2*2

    elif nset==63:
        goalString=["(on green table2) (on blue green)", "(on blue green) (on green table2)"]
        if getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green table2) (place pr2L blue green)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue green) (place pr2R green table2)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R green) (place pr2R green table2) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue green) (place pr2L green table2)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green table2) (place pr2R blue green)",
                        "(grasp pr2R blue) (place pr2R blue green) (grasp pr2R green) (place pr2R green table2)",
                        "(grasp pr2R blue) (place pr2R blue green) (grasp pr2L green) (place pr2L green table2)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue green) (place pr2L green table2)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table2) (place pr2R blue green)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L green) (place pr2L green table2) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2L blue) (place pr2L blue green) (grasp pr2R green) (place pr2R green table2)",
                        "(grasp pr2L blue) (place pr2L blue green) (grasp pr2L green) (place pr2L green table2)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue green) (place pr2R green table2)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table2) (place pr2L blue green)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2R blue) (place pr2R blue green)"]
            numLoops=2*2*2

    elif nset==64:
        goalString=["(on green red) (held blue)", "(held blue) (on green red)"]
        solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green red)",
                    "(grasp pr2R green) (place pr2R green red) (grasp pr2L blue)",
                    "(grasp pr2R green) (place pr2R green red) (grasp pr2R blue)",
                    "(grasp pr2R blue) (grasp pr2L green) (place pr2L green red)",
                    "(grasp pr2L green) (grasp pr2R blue) (place pr2L green red)",
                    "(grasp pr2L green) (place pr2L green red) (grasp pr2R blue)",
                    "(grasp pr2L green) (place pr2L green red) (grasp pr2L blue)",
                    "(grasp pr2L blue) (grasp pr2R green) (place pr2R green red)"]
        numLoops=2*8*3

    elif nset==65:
        goalString=["(on green red) (on blue table1)", "(on blue table1) (on green red)"]
        if getEnvInfo(nenv,"blue") in [0,2]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green red) (place pr2L blue table1)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (place pr2R green red)",
                        "(grasp pr2R green) (place pr2R green red) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R green) (place pr2R green red) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (place pr2L green red)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green red) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green red) (place pr2R blue table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (place pr2L green red)",
                        "(grasp pr2L green) (place pr2L green red) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L green) (place pr2L green red) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (place pr2R green red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green red) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green red)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"blue") in [0,1]:
            solutions=["(grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L green) (place pr2L green red)"]
            numLoops=2*2*2

    elif nset==66:
        goalString=["(on green red) (on blue table2)", "(on blue table2) (on green red)"]
        if getEnvInfo(nenv,"blue") in [0,1]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green red) (place pr2L blue table2)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table2) (place pr2R green red)",
                        "(grasp pr2R green) (place pr2R green red) (grasp pr2L blue) (place pr2L blue table2)",
                        "(grasp pr2R green) (place pr2R green red) (grasp pr2R blue) (place pr2R blue table2)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table2) (place pr2L green red)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green red) (place pr2R blue table2)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green red) (place pr2R blue table2)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table2) (place pr2L green red)",
                        "(grasp pr2L green) (place pr2L green red) (grasp pr2L blue) (place pr2L blue table2)",
                        "(grasp pr2L green) (place pr2L green red) (grasp pr2R blue) (place pr2R blue table2)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table2) (place pr2R green red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green red) (place pr2L blue table2)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2L green) (place pr2L green red)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"blue") in [0,2]:
            solutions=["(grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L green) (place pr2L green red)"]
            numLoops=2*2*2

    elif nset==67:
        goalString=["(on green red) (on blue red)", "(on blue red) (on green red)"]
        solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green red) (place pr2L blue red)",
                    "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue red) (place pr2R green red)",
                    "(grasp pr2R green) (place pr2R green red) (grasp pr2L blue) (place pr2L blue red)",
                    "(grasp pr2R green) (place pr2R green red) (grasp pr2R blue) (place pr2R blue red)",
                    "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue red) (place pr2L green red)",
                    "(grasp pr2R blue) (grasp pr2L green) (place pr2L green red) (place pr2R blue red)",
                    "(grasp pr2R blue) (place pr2R blue red) (grasp pr2R green) (place pr2R green red)",
                    "(grasp pr2R blue) (place pr2R blue red) (grasp pr2L green) (place pr2L green red)",
                    "(grasp pr2L green) (grasp pr2R blue) (place pr2L green red) (place pr2R blue red)",
                    "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue red) (place pr2L green red)",
                    "(grasp pr2L green) (place pr2L green red) (grasp pr2L blue) (place pr2L blue red)",
                    "(grasp pr2L green) (place pr2L green red) (grasp pr2R blue) (place pr2R blue red)",
                    "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue red) (place pr2R green red)",
                    "(grasp pr2L blue) (grasp pr2R green) (place pr2R green red) (place pr2L blue red)",
                    "(grasp pr2L blue) (place pr2L blue red) (grasp pr2R green) (place pr2R green red)",
                    "(grasp pr2L blue) (place pr2L blue red) (grasp pr2L green) (place pr2L green red)"]
        numLoops=2*16*4

    elif nset==68:
        goalString=["(on green red) (on blue green)", "(on blue green) (on green red)"]
        solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green red) (place pr2L blue green)",
                    "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue green) (place pr2R green red)",
                    "(grasp pr2R green) (place pr2R green red) (grasp pr2R blue) (place pr2R blue green)",
                    "(grasp pr2R green) (place pr2R green red) (grasp pr2L blue) (place pr2L blue green)",
                    "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue green) (place pr2L green red)",
                    "(grasp pr2R blue) (grasp pr2L green) (place pr2L green red) (place pr2R blue green)",
                    "(grasp pr2R blue) (place pr2R blue green) (grasp pr2R green) (place pr2R green red)",
                    "(grasp pr2R blue) (place pr2R blue green) (grasp pr2L green) (place pr2L green red)",
                    "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue green) (place pr2L green red)",
                    "(grasp pr2L green) (grasp pr2R blue) (place pr2L green red) (place pr2R blue green)",
                    "(grasp pr2L green) (place pr2L green red) (grasp pr2R blue) (place pr2R blue green)",
                    "(grasp pr2L green) (place pr2L green red) (grasp pr2L blue) (place pr2L blue green)",
                    "(grasp pr2L blue) (grasp pr2R green) (place pr2R green red) (place pr2L blue green)",
                    "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue green) (place pr2R green red)",
                    "(grasp pr2L blue) (place pr2L blue green) (grasp pr2R green) (place pr2R green red)",
                    "(grasp pr2L blue) (place pr2L blue green) (grasp pr2L green) (place pr2L green red)"]
        numLoops=2*16*4

    elif nset==69:
        goalString=["(on green blue) (held blue)", "(held blue) (on green blue)"]
        solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green blue)",
                    "(grasp pr2R green) (place pr2R green blue) (grasp pr2R blue)",
                    "(grasp pr2R green) (place pr2R green blue) (grasp pr2L blue)",
                    "(grasp pr2R blue) (grasp pr2L green) (place pr2L green blue)",
                    "(grasp pr2L green) (grasp pr2R blue) (place pr2L green blue)",
                    "(grasp pr2L green) (place pr2L green blue) (grasp pr2R blue)",
                    "(grasp pr2L green) (place pr2L green blue) (grasp pr2L blue)",
                    "(grasp pr2L blue) (grasp pr2R green) (place pr2R green blue)"]
        numLoops=2*8*3

    elif nset==70:
        goalString=["(on green blue) (on blue table1)", "(on blue table1) (on green blue)"]
        if getEnvInfo(nenv,"blue") in [0,2]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue table1)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (place pr2R green blue)",
                        "(grasp pr2R green) (place pr2R green blue) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R green) (place pr2R green blue) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (place pr2L green blue)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green blue) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (place pr2L green blue)",
                        "(grasp pr2L green) (place pr2L green blue) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L green) (place pr2L green blue) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (place pr2R green blue)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green blue) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green blue)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"blue") in [0,1]:
            solutions=["(grasp pr2R green) (place pr2R green blue)",
                    "(grasp pr2L green) (place pr2L green blue)"]
            numLoops=2*2*2

    elif nset==71:
        goalString=["(on green blue) (on blue table2)", "(on blue table2) (on green blue)"]
        if getEnvInfo(nenv,"blue") in [0,1]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue table2)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table2) (place pr2R green blue)",
                        "(grasp pr2R green) (place pr2R green blue) (grasp pr2R blue) (place pr2R blue table2)",
                        "(grasp pr2R green) (place pr2R green blue) (grasp pr2L blue) (place pr2L blue table2)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table2) (place pr2L green blue)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green blue) (place pr2R blue table2)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R blue) (place pr2R blue table2) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue table2)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table2) (place pr2L green blue)",
                        "(grasp pr2L green) (place pr2L green blue) (grasp pr2R blue) (place pr2R blue table2)",
                        "(grasp pr2L green) (place pr2L green blue) (grasp pr2L blue) (place pr2L blue table2)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table2) (place pr2R green blue)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green blue) (place pr2L blue table2)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L blue) (place pr2L blue table2) (grasp pr2L green) (place pr2L green blue)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"blue") in [0,2]:
            solutions=["(grasp pr2R green) (place pr2R green blue)",
                    "(grasp pr2L green) (place pr2L green blue)"]
            numLoops=2*2*2

    elif nset==72:
        goalString=["(on green blue) (on blue red)", "(on blue red) (on green blue)"]
        solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue red)",
                    "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue red) (place pr2R green blue)",
                    "(grasp pr2R green) (place pr2R green blue) (grasp pr2R blue) (place pr2R blue red)",
                    "(grasp pr2R green) (place pr2R green blue) (grasp pr2L blue) (place pr2L blue red)",
                    "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue red) (place pr2L green blue)",
                    "(grasp pr2R blue) (grasp pr2L green) (place pr2L green blue) (place pr2R blue red)",
                    "(grasp pr2R blue) (place pr2R blue red) (grasp pr2R green) (place pr2R green blue)",
                    "(grasp pr2R blue) (place pr2R blue red) (grasp pr2L green) (place pr2L green blue)",
                    "(grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue red)",
                    "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue red) (place pr2L green blue)",
                    "(grasp pr2L green) (place pr2L green blue) (grasp pr2R blue) (place pr2R blue red)",
                    "(grasp pr2L green) (place pr2L green blue) (grasp pr2L blue) (place pr2L blue red)",
                    "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue red) (place pr2R green blue)",
                    "(grasp pr2L blue) (grasp pr2R green) (place pr2R green blue) (place pr2L blue red)",
                    "(grasp pr2L blue) (place pr2L blue red) (grasp pr2R green) (place pr2R green blue)",
                    "(grasp pr2L blue) (place pr2L blue red) (grasp pr2L green) (place pr2L green blue)"]
        numLoops=2*16*4

    else:
        input("\nNot implemented: set "+str(nset).zfill(3))

    return solutions, goalString, numLoops


def main():

    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--env', type=int, default=1)
    parser.add_argument('--set', type=int, default=1)
    args = parser.parse_args()
    #verbose=args.verbose
    nenv=args.env
    nset=args.set
    
    if nenv in range(1,numEnv+1) and nset in range(1,numSets+1):
        solutions, _, _ = getData(nenv=nenv,nset=nset)
        print(solutions)
        print(getEnvInfo(nenv,"red"), getEnvInfo(nenv,"green"), getEnvInfo(nenv,"blue"))
    else:
        print("Not implemented nenv: "+str(nenv)+" nset: "+str(nset)+"\n\n")
        
        #num=1
        #for i in range(0,len(Sets),2):
        #        print("elif nset=="+str(num)+":")
        #        print('\tgoalString=["'+Sets[i]+'", "'+Sets[i+1]+'"]')
        #        print()
        #        num+=1

    

if __name__ == "__main__":
    main()