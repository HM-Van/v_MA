import numpy as np
# Information for all init configurations: obj on which table

# 0 table, 1 red, 2 green, 3 blue
envInfo=x = np.array([[0, 0, 0], #1 r g b
                    [0, 0, 0], #2
                    [0, 0, 0], #3
                    [0, 0, 0], #4
                    [0, 1, 0], #5
                    [0, 1, 0], #6
                    [0, 1, 0], #7
                    [0, 1, 0], #8
                    [0, 0, 1], #9
                    [0, 0, 1], #10
                    [0, 0, 1], #11
                    [0, 0, 1], #12
                    [0, 0, 2], #13
                    [0, 0, 2], #14
                    [0, 0, 2], #15
                    [0, 0, 2], #16
                    [0, 3, 0], #17
                    [0, 3, 0], #18
                    [0, 3, 0], #19
                    [0, 3, 0], #20
                    [2, 0, 0], #21
                    [2, 0, 0], #22
                    [2, 0, 0], #23
                    [2, 0, 0], #24
                    [3, 0, 0], #25
                    [3, 0, 0], #26
                    [3, 0, 0], #27
                    [3, 0, 0], #28
                    [0, 1, 2], #29
                    [0, 1, 2], #30
                    [0, 1, 2], #31
                    [0, 1, 2], #32
                    [0, 3, 1], #33
                    [0, 3, 1], #34
                    [0, 3, 1], #35
                    [0, 3, 1], #36
                    [3, 0, 2], #37
                    [3, 0, 2], #38
                    [3, 0, 2], #39
                    [3, 0, 2], #40
                    [2, 0, 1], #41
                    [2, 0, 1], #42
                    [2, 0, 1], #43
                    [2, 0, 1], #44
                    [3, 1, 0], #45
                    [3, 1, 0], #46
                    [3, 1, 0], #47
                    [3, 1, 0], #48
                    [2, 3, 0], #49
                    [2, 3, 0], #50
                    [2, 3, 0], #51
                    [2, 3, 0], #52
                    ],
                    np.int16)

test=["(held red)", "(on red table1)", "(on red green)", "(on red blue)",
        "(held green)", "(on green table1)", "(on green red)", "(on green blue)",
        "(held blue)", "(on blue table1)", "(on blue red)", "(on blue green)"]

numSets=72
numEnv=99
# objectives consisting of 2 goal formulations
Sets=[
        "(held red) (held green)" , "(held green) (held red)",#1
        "(held red) (on green table1)" , "(on green table1) (held red)",
        "(held red) (on green red)" , "(on green red) (held red)",
        "(held red) (on green blue)" , "(on green blue) (held red)",
        "(held red) (held blue)" , "(held blue) (held red)",#5
        "(held red) (on blue table1)" , "(on blue table1) (held red)",
        "(held red) (on blue red)" , "(on blue red) (held red)",
        "(held red) (on blue green)" , "(on blue green) (held red)",
        "(on red table1) (held green)" , "(held green) (on red table1)",
        "(on red table1) (on green table1)" , "(on green table1) (on red table1)",#10
        "(on red table1) (on green red)" , "(on green red) (on red table1)",
        "(on red table1) (on green blue)" , "(on green blue) (on red table1)",
        "(on red table1) (held blue)" , "(held blue) (on red table1)",
        "(on red table1) (on blue table1)" , "(on blue table1) (on red table1)",
        "(on red table1) (on blue red)" , "(on blue red) (on red table1)",#15
        "(on red table1) (on blue green)" , "(on blue green) (on red table1)",
        "(on red green) (held green)" , "(held green) (on red green)",
        "(on red green) (on green table1)" , "(on green table1) (on red green)",
        "(on red green) (on green blue)" , "(on green blue) (on red green)",
        "(on red green) (held blue)" , "(held blue) (on red green)",#20
        "(on red green) (on blue table1)" , "(on blue table1) (on red green)",
        "(on red green) (on blue red)" , "(on blue red) (on red green)",
        "(on red blue) (held green)" , "(held green) (on red blue)",
        "(on red blue) (on green table1)" , "(on green table1) (on red blue)",
        "(on red blue) (on green red)" , "(on green red) (on red blue)",#25
        "(on red blue) (held blue)" , "(held blue) (on red blue)",
        "(on red blue) (on blue table1)" , "(on blue table1) (on red blue)",
        "(on red blue) (on blue green)" , "(on blue green) (on red blue)",
        "(held green) (held blue)" , "(held blue) (held green)",
        "(held green) (on blue table1)" , "(on blue table1) (held green)",#30
        "(held green) (on blue red)" , "(on blue red) (held green)",
        "(on green table1) (held blue)" , "(held blue) (on green table1)",
        "(on green table1) (on blue table1)" , "(on blue table1) (on green table1)",
        "(on green table1) (on blue red)" , "(on blue red) (on green table1)",
        "(on green table1) (on blue green)" , "(on blue green) (on green table1)",#35
        "(on green red) (held blue)" , "(held blue) (on green red)",
        "(on green red) (on blue table1)" , "(on blue table1) (on green red)",
        "(on green red) (on blue green)" , "(on blue green) (on green red)",
        "(on green blue) (held blue)" , "(held blue) (on green blue)",
        "(on green blue) (on blue table1)" , "(on blue table1) (on green blue)",#40
        "(on green blue) (on blue red)" , "(on blue red) (on green blue)",
        "(held green) (on blue green)" , "(on blue green) (held green)"
        ]

def getEnvInfo(env,key):
    if key=="r" or key=="red" or key==0 or key=="0":
        return envInfo[env-1,0]
    if key=="g" or key=="green" or key==1 or key=="1":
        return envInfo[env-1,1]
    if key=="b" or key=="blue" or key==2 or key=="2":
        return envInfo[env-1,2]
    else:
        print("Not defined: ", key)
        return 0

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
        if getEnvInfo(nenv,"red") in [0]:
            solutions=[]
            numLoops=0
        elif getEnvInfo(nenv, "red") in [1,2,3]:
            solutions=["(grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L red) (place pr2L red table1)"]
            numLoops=1*2*2
        

    elif nset==3:
        goalString=["(on red green) (on red green)"] #rg
        if getEnvInfo(nenv,"red") in [2]:
            solutions=[]
            numLoops=0
        
        elif getEnvInfo(nenv,"green") in [1] and getEnvInfo(nenv,"blue") in [2]: # bgr
            solutions=["(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (place pr2L red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2L red green) (place pr2R green table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (place pr2L red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2R red green) (place pr2L green table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red green)",
                        
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (place pr2L red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2L red green) (place pr2R green table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (place pr2L red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2R red green) (place pr2L green table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red green)"]
            numLoops=1*24*6

        elif getEnvInfo(nenv,"blue") in [2]:
            solutions=[ "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red green)"]
            numLoops=1*4*4
        
        elif getEnvInfo(nenv,"green") in [0] or (getEnvInfo(nenv,"green") in [3] and getEnvInfo(nenv,"blue") in [0]): # gt or gbt 
            solutions=["(grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L red) (place pr2L red green)"]
            numLoops=1*2*2
        
        elif getEnvInfo(nenv,"green") in [1] or (getEnvInfo(nenv,"green") in [3] and getEnvInfo(nenv,"blue") in [1]): # gr or gbr 
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green table1)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red green)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (place pr2L red green)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red green) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (place pr2L red green)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (place pr2R red green)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red green) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red green)"]
            numLoops=1*12*4

    elif nset==4:
        goalString=["(on red blue) (on red blue)"] # rb
        if getEnvInfo(nenv,"red") in [3]:
            solutions=[]
            numLoops=0

        elif getEnvInfo(nenv,"blue") in [1] and getEnvInfo(nenv,"green") in [3]: # gbr
            solutions=["(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (place pr2R red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (place pr2L red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2L red blue) (place pr2R blue table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (place pr2L red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (place pr2R red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2R red blue) (place pr2L blue table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red blue)",
                        
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (place pr2R red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (place pr2L red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2L red blue) (place pr2R blue table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (place pr2L red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (place pr2R red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2R red blue) (place pr2L blue table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red blue)"]
            numLoops=1*24*6
        
        elif getEnvInfo(nenv,"green") in [3]:
            solutions=[ "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red blue)"]
            numLoops=1*4*4

        elif getEnvInfo(nenv,"blue") in [0] or (getEnvInfo(nenv,"blue") in [2] and getEnvInfo(nenv,"green") in [0]): # bt or bgt 
            solutions=["(grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L red) (place pr2L red blue)"]
            numLoops=1*2*2
        
        elif getEnvInfo(nenv,"blue") in [1] or (getEnvInfo(nenv,"blue") in [2] and getEnvInfo(nenv,"green") in [1]): # br or bgr 
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue table1)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (place pr2R red blue)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (place pr2L red blue)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red blue) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (place pr2L red blue)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (place pr2R red blue)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red blue) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red blue)"]
            numLoops=1*12*4

    elif nset==5:
        goalString=["(held green) (held green)"]
        solutions=["(grasp pr2R green)",
                    "(grasp pr2L green)"]
        numLoops=1*2*1
    
    elif nset==6:
        goalString=["(on green table1) (on green table1)"]
        if getEnvInfo(nenv,"green") in [0]:
            solutions=[]
            numLoops=0
        
        elif getEnvInfo(nenv, "green") in [1,2,3]:
            solutions=["(grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L green) (place pr2L green table1)"]
            numLoops=1*2*2

    elif nset==7:
        goalString=["(on green red) (on green red)"] #gr

        if getEnvInfo(nenv,"green") in [1]:
            solutions=[]
            numLoops=0

        elif getEnvInfo(nenv,"red") in [2] and getEnvInfo(nenv,"blue") in [1]: # brg
            solutions=["(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2R green red) (place pr2L red table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2L red table1) (place pr2R green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2R red table1) (place pr2L green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2L green red) (place pr2R red table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2L green red) (place pr2R red table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2R red table1) (place pr2L green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2L red table1) (place pr2R green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2R green red) (place pr2L red table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green red)",
                        
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2R green red) (place pr2L red table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2L red table1) (place pr2R green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2R red table1) (place pr2L green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2L green red) (place pr2R red table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2L green red) (place pr2R red table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2R red table1) (place pr2L green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2L red table1) (place pr2R green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2R green red) (place pr2L red table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green red)"]
            numLoops=1*24*6
        
        elif getEnvInfo(nenv,"blue") in [1]:
            solutions=[ "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green red)"]
            numLoops=1*4*4

        elif getEnvInfo(nenv,"red") in [0] or (getEnvInfo(nenv,"red") in [3] and getEnvInfo(nenv,"blue") in [0]): # rt or rbt 
            solutions=["(grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L green) (place pr2L green red)"]
            numLoops=1*2*2
        
        elif getEnvInfo(nenv,"red") in [2] or (getEnvInfo(nenv,"red") in [3] and getEnvInfo(nenv,"blue") in [2]): # rg or rbg
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2R green red) (place pr2L red table1)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red table1) (place pr2R green red)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red table1) (place pr2L green red)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green red) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green red) (place pr2R red table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table1) (place pr2L green red)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table1) (place pr2R green red)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green red) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green red)"]
            numLoops=1*12*4

    elif nset==8:
        goalString=["(on green blue) (on green blue)"]
        if getEnvInfo(nenv,"green") in [3]:
            solutions=[]
            numLoops=0

        elif getEnvInfo(nenv,"blue") in [2] and getEnvInfo(nenv,"red") in [3]: # rbg
            solutions=["(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (place pr2R green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (place pr2L green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2L green blue) (place pr2R blue table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (place pr2L green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (place pr2R green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2R green blue) (place pr2L blue table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green blue)",
                        
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (place pr2R green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (place pr2L green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2L green blue) (place pr2R blue table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (place pr2L green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (place pr2R green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2R green blue) (place pr2L blue table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green blue)"]
            numLoops=1*24*6

        elif getEnvInfo(nenv,"red") in [3]:
            solutions=[ "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green blue)"]
            numLoops=1*4*4

        elif getEnvInfo(nenv,"blue") in [0] or (getEnvInfo(nenv,"blue") in [1] and getEnvInfo(nenv,"red") in [0]): # bt or brt 
            solutions=["(grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L green) (place pr2L green blue)"]
            numLoops=1*2*2
        
        elif getEnvInfo(nenv,"blue") in [2] or (getEnvInfo(nenv,"blue") in [1] and getEnvInfo(nenv,"red") in [2]): # bg or brg
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue table1)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (place pr2R green blue)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (place pr2L green blue)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green blue) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (place pr2L green blue)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (place pr2R green blue)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green blue) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green blue)"]
            numLoops=1*12*4

    elif nset==9:
        goalString=["(held blue) (held blue)"]
        solutions=["(grasp pr2R blue)",
                    "(grasp pr2L blue)"]
        numLoops=1*2*1
    
    elif nset==10:
        goalString=["(on blue table1) (on blue table1)"]
        if getEnvInfo(nenv,"blue") in [0]:
            solutions=[]
            numLoops=0

        elif getEnvInfo(nenv, "blue") in [1,2,3]:
            solutions=["(grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1)"]
            numLoops=1*2*2

    elif nset==11:
        goalString=["(on blue red) (on blue red)"] #br
        if getEnvInfo(nenv,"blue") in [1]:
            solutions=[]
            numLoops=0
        
        elif getEnvInfo(nenv,"red") in [3] and getEnvInfo(nenv,"green") in [1]: # grb
            solutions=["(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2R blue red) (place pr2L red table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2L red table1) (place pr2R blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2R red table1) (place pr2L blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2L blue red) (place pr2R red table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2L blue red) (place pr2R red table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2R red table1) (place pr2L blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2L red table1) (place pr2R blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2R blue red) (place pr2L red table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue red)",
                        
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2R blue red) (place pr2L red table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2L red table1) (place pr2R blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2R red table1) (place pr2L blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2L blue red) (place pr2R red table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2L blue red) (place pr2R red table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2R red table1) (place pr2L blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2L red table1) (place pr2R blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2R blue red) (place pr2L red table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue red)"]
            numLoops=1*24*6
        
        elif getEnvInfo(nenv,"green") in [1]:
            solutions=[ "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue red)"]
            numLoops=1*4*4
        
        elif getEnvInfo(nenv,"red") in [0] or (getEnvInfo(nenv,"red") in [2] and getEnvInfo(nenv,"green") in [0]): # rt or rgt 
            solutions=["(grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L blue) (place pr2L blue red)"]
            numLoops=1*2*2
        
        elif getEnvInfo(nenv,"red") in [3] or (getEnvInfo(nenv,"red") in [2] and getEnvInfo(nenv,"green") in [3]): # rb or rgb
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2R blue red) (place pr2L red table1)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red table1) (place pr2R blue red)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red table1) (place pr2L blue red)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue red) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue red) (place pr2R red table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table1) (place pr2L blue red)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table1) (place pr2R blue red)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue red) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue red)"]
            numLoops=1*12*4

    elif nset==12:
        goalString=["(on blue green) (on blue green)"] #bg
        if getEnvInfo(nenv,"blue") in [2]:
            solutions=[]
            numLoops=0

        elif getEnvInfo(nenv,"green") in [3] and getEnvInfo(nenv,"red") in [2]: # rgb
            solutions=["(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2R blue green) (place pr2L green table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2L green table1) (place pr2R blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2R green table1) (place pr2L blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2L blue green) (place pr2R green table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2L blue green) (place pr2R green table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2R green table1) (place pr2L blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2L green table1) (place pr2R blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2R blue green) (place pr2L green table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue green)",
                        
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2R blue green) (place pr2L green table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2L green table1) (place pr2R blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2R green table1) (place pr2L blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2L blue green) (place pr2R green table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2L blue green) (place pr2R green table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2R green table1) (place pr2L blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2L green table1) (place pr2R blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2R blue green) (place pr2L green table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue green)"]
            numLoops=1*24*6

        elif getEnvInfo(nenv,"red") in [2]:
            solutions=[ "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue green)"]
            numLoops=1*4*4
        
        elif getEnvInfo(nenv,"green") in [0] or (getEnvInfo(nenv,"green") in [1] and getEnvInfo(nenv,"red") in [0]): # gt or grt 
            solutions=["(grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L blue) (place pr2L blue green)"]
            numLoops=1*2*2
        
        elif getEnvInfo(nenv,"green") in [3] or (getEnvInfo(nenv,"green") in [1] and getEnvInfo(nenv,"red") in [3]): # gb or grb
            solutions=["(grasp pr2R blue) (grasp pr2L green) (place pr2R blue green) (place pr2L green table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green table1) (place pr2R blue green)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2R green table1) (place pr2L blue green)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue green) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue green) (place pr2R green table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table1) (place pr2L blue green)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table1) (place pr2R blue green)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue green) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue green)"]
            numLoops=1*12*4

    return solutions, goalString, numLoops

# Get solutions for objectives consisting of 2 goal formulations
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
        if getEnvInfo(nenv,"green") in [1,2,3]:
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red)"]
            numLoops=2*8*3
        elif getEnvInfo(nenv,"green") in [0]:
            solutions=["(grasp pr2R red)","(grasp pr2L red)"]
            numLoops=2*2*1

    elif nset==3:
        goalString=["(held red) (on green red)", "(on green red) (held red)"]
        if getEnvInfo(nenv,"red") in [3] and getEnvInfo(nenv,"blue") in [2] or getEnvInfo(nenv,"blue") in [1]:
            solutions=["(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2R green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2L green red)",
                        
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2R green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2L green red)"]
            numLoops=2*8*5

        elif getEnvInfo(nenv,"red") in [2]:
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green red)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green red)"]
            numLoops=2*4*3
        
        elif getEnvInfo(nenv,"green") in [0,2,3]:
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green red)",
                        "(grasp pr2R green) (place pr2R green red) (grasp pr2R red)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green red)",
                        "(grasp pr2L green) (place pr2L green red) (grasp pr2R red)"]
            numLoops=2*6*3
        elif getEnvInfo(nenv,"green") in [1]:
            solutions=["(grasp pr2R red)","(grasp pr2L red)"]
            numLoops=2*2*1

    elif nset==4:
        goalString=["(held red) (on green blue)", "(on green blue) (held red)"]

        if getEnvInfo(nenv,"blue") in [2] and getEnvInfo(nenv,"red") in [3]: # rbg
            solutions=[ "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (grasp pr2L green) (place pr2L green blue)"]
            numLoops=2*4*5

        elif getEnvInfo(nenv,"red") in [3]:
            solutions=[ "(grasp pr2R red) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green blue)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green blue)"]
            numLoops=2*4*3

        elif getEnvInfo(nenv,"blue") in [0] or (getEnvInfo(nenv,"blue") in [1] and getEnvInfo(nenv,"red") in [0]): # bt or brt 
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2R green blue)",
                        "(grasp pr2R green) (place pr2R green blue) (grasp pr2R red)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green blue)",
                        "(grasp pr2L green) (place pr2L green blue) (grasp pr2R red)"]
            numLoops=2*4*3
        
        elif getEnvInfo(nenv,"blue") in [2] or (getEnvInfo(nenv,"blue") in [1] and getEnvInfo(nenv,"red") in [2]): # bg or brg
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue table1) (grasp pr2R red)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (place pr2R green blue) (grasp pr2R red)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (place pr2L green blue) (grasp pr2R red)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green blue) (place pr2R blue table1) (grasp pr2R red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green blue) (grasp pr2R red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green blue) (grasp pr2R red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue table1) (grasp pr2R red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (place pr2L green blue) (grasp pr2R red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (place pr2R green blue) (grasp pr2R red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green blue) (place pr2L blue table1) (grasp pr2R red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green blue) (grasp pr2R red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green blue) (grasp pr2R red)",
                        
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue table1) (grasp pr2L red)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (place pr2R green blue) (grasp pr2L red)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (place pr2L green blue) (grasp pr2L red)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green blue) (place pr2R blue table1) (grasp pr2L red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green blue) (grasp pr2L red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green blue) (grasp pr2L red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue table1) (grasp pr2L red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (place pr2L green blue) (grasp pr2L red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (place pr2R green blue) (grasp pr2L red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green blue) (place pr2L blue table1) (grasp pr2L red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green blue) (grasp pr2L red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green blue) (grasp pr2L red)"]
            numLoops=2*24*5
        elif getEnvInfo(nenv,"green") in [3]:
            solutions=["(grasp pr2R red)","(grasp pr2L red)"]
            numLoops=2*2*1

    elif nset==5:
        goalString=["(held red) (held blue)", "(held blue) (held red)"]
        goalString=["(held red) (held blue)", "(held blue) (held red)"]
        solutions=["(grasp pr2R red) (grasp pr2L blue)",
                    "(grasp pr2R blue) (grasp pr2L red)",
                    "(grasp pr2L red) (grasp pr2R blue)",
                    "(grasp pr2R blue) (grasp pr2L red)"]
        numLoops=2*4*2

    elif nset==6:
        goalString=["(held red) (on blue table1)", "(on blue table1) (held red)"]
        if getEnvInfo(nenv,"blue") in [1,2,3]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red)"]
            numLoops=2*8*3
        elif getEnvInfo(nenv,"blue") in [0]:
            solutions=["(grasp pr2R red)","(grasp pr2L red)"]
            numLoops=2*2*1

    elif nset==7:
        goalString=["(held red) (on blue red)", "(on blue red) (held red)"]
        if getEnvInfo(nenv,"red") in [2] and getEnvInfo(nenv,"green") in [3] or getEnvInfo(nenv,"green") in [1]:
            solutions=["(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2R blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2L blue red)",
                        
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2R blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2L blue red)"]
            numLoops=2*8*5
        elif getEnvInfo(nenv,"red") in [3]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue red)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue red)"]
            numLoops=2*4*3

        elif getEnvInfo(nenv,"blue") in [0,2,3]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue red)",
                        "(grasp pr2R blue) (place pr2R blue red) (grasp pr2R red)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue red)",
                        "(grasp pr2L blue) (place pr2L blue red) (grasp pr2R red)"]
            numLoops=2*6*3
        elif getEnvInfo(nenv,"blue") in [1]:
            solutions=["(grasp pr2R red)","(grasp pr2L red)"]
            numLoops=2*2*1

    elif nset==8:
        goalString=["(held red) (on blue green)", "(on blue green) (held red)"]

        if getEnvInfo(nenv,"green") in [3] and getEnvInfo(nenv,"red") in [2]: # rgb
            solutions=[ "(grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue green)"]
            numLoops=2*4*5

        elif getEnvInfo(nenv,"red") in [2]:
            solutions=[ "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue green)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue green)"]
            numLoops=2*4*3
          
        elif getEnvInfo(nenv,"green") in [0] or (getEnvInfo(nenv,"green") in [1] and getEnvInfo(nenv,"red") in [0]): # gt or grt 
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2R blue green)",
                        "(grasp pr2R blue) (place pr2R blue green) (grasp pr2R red)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue green)",
                        "(grasp pr2L blue) (place pr2L blue green) (grasp pr2R red)"]
            numLoops=2*4*3
        
        elif getEnvInfo(nenv,"green") in [3] or (getEnvInfo(nenv,"green") in [1] and getEnvInfo(nenv,"red") in [3]): # gb or grb
            solutions=["(grasp pr2R blue) (grasp pr2L green) (place pr2R blue green) (place pr2L green table1) (grasp pr2R red)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green table1) (place pr2R blue green) (grasp pr2R red)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2R green table1) (place pr2L blue green) (grasp pr2R red)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue green) (place pr2R green table1) (grasp pr2R red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue green) (grasp pr2R red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue green) (grasp pr2R red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue green) (place pr2R green table1) (grasp pr2R red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table1) (place pr2L blue green) (grasp pr2R red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table1) (place pr2R blue green) (grasp pr2R red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue green) (place pr2L green table1) (grasp pr2R red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue green) (grasp pr2R red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue green) (grasp pr2R red)",
                        
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue green) (place pr2L green table1) (grasp pr2L red)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green table1) (place pr2R blue green) (grasp pr2L red)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2R green table1) (place pr2L blue green) (grasp pr2L red)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue green) (place pr2R green table1) (grasp pr2L red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue green) (grasp pr2L red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue green) (grasp pr2L red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue green) (place pr2R green table1) (grasp pr2L red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table1) (place pr2L blue green) (grasp pr2L red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table1) (place pr2R blue green) (grasp pr2L red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue green) (place pr2L green table1) (grasp pr2L red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue green) (grasp pr2L red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue green) (grasp pr2L red)"]
            numLoops=2*24*5
        elif getEnvInfo(nenv,"blue") in [2]:
            solutions=["(grasp pr2R red)","(grasp pr2L red)"]
            numLoops=2*2*1

    elif nset==9:
        goalString=["(on red table1) (held green)", "(held green) (on red table1)"]
        if getEnvInfo(nenv,"red") in [1,2,3]:
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green)"]
            numLoops=2*8*3
        elif getEnvInfo(nenv,"red") in [0]:
            solutions=["(grasp pr2R green)","(grasp pr2L green)"]
            numLoops=2*2*1

    elif nset==10:
        goalString=["(on red table1) (on green table1)", "(on green table1) (on red table1)"]
        if getEnvInfo(nenv,"red") in [0] and getEnvInfo(nenv,"green") in [0]:
            solutions=[]
            numLoops=0
        elif getEnvInfo(nenv,"red") in [1,2,3] and getEnvInfo(nenv,"green") in [1,2,3]:
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
        elif getEnvInfo(nenv,"red") in [1,2,3] and getEnvInfo(nenv,"green") in [0]:
            solutions=["(grasp pr2L red) (place pr2L red table1)",
                    "(grasp pr2R red) (place pr2R red table1)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"red") in [0] and getEnvInfo(nenv,"green") in [1,2,3]:
            solutions=["(grasp pr2L green) (place pr2L green table1)",
                    "(grasp pr2R green) (place pr2R green table1)"]
            numLoops=2*2*2
        

    elif nset==11:
        goalString=["(on red table1) (on green red)", "(on green red) (on red table1)"]

        if getEnvInfo(nenv,"red") in [0] and getEnvInfo(nenv,"green") in [1]:
            solutions=[]
            numLoops=0

        elif getEnvInfo(nenv,"red") in [3] and getEnvInfo(nenv,"blue") in [2] or getEnvInfo(nenv,"blue") in [1] and getEnvInfo(nenv,"red") in [2]:
            solutions=["(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2R red table1) (place pr2L green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2L green red) (place pr2R red table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green red)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green red)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2R green red) (place pr2L red table1)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2L red table1) (place pr2R green red)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2L red table1) (place pr2R green red)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2R green red) (place pr2L red table1)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green red)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green red)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2L green red) (place pr2R red table1)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2R red table1) (place pr2L green red)",
                        
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2R red table1) (place pr2L green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2L green red) (place pr2R red table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green red)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green red)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2R green red) (place pr2L red table1)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2L red table1) (place pr2R green red)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2L red table1) (place pr2R green red)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2R green red) (place pr2L red table1)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green red)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green red)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2L green red) (place pr2R red table1)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2R red table1) (place pr2L green red)"]
            numLoops=2*24*6

        elif getEnvInfo(nenv,"blue") in [1] and getEnvInfo(nenv,"red") in [0]:
            solutions=[ "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green red)"]
            numLoops=2*4*4
            
        elif getEnvInfo(nenv,"red") in [2]:
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red table1) (place pr2L green red)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green red) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green red)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green red)", 
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green red) (place pr2L red table1)", 
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red table1) (place pr2R green red)", 
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table1) (place pr2R green red)", 
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green red) (place pr2L red table1)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green red)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green red)", 
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green red) (place pr2R red table1)", 
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table1) (place pr2L green red)"]
            numLoops=2*12*4

        elif getEnvInfo(nenv,"red") in [1,3] and getEnvInfo(nenv,"green") in [0,2,3]:
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
        elif getEnvInfo(nenv,"red") in [0] and getEnvInfo(nenv,"green") in [0,2,3]:
            solutions=["(grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2R green) (place pr2R green red)"]
            numLoops=2*2*2

        elif getEnvInfo(nenv,"red") in [1,3] and getEnvInfo(nenv,"green") in [1]:
            solutions=["(grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L red) (place pr2L red table1)"]
            numLoops=2*2*2
    

    elif nset==12:
        goalString=["(on red table1) (on green blue)", "(on green blue) (on red table1)"]
        if getEnvInfo(nenv,"red") in [0] and getEnvInfo(nenv,"green") in [3]:
            solutions=[]
            numLoops=0
        elif getEnvInfo(nenv,"blue") in [1] and getEnvInfo(nenv,"red") in [2] or getEnvInfo(nenv,"blue") in [2] and getEnvInfo(nenv,"red") in [1,2,3]: # brg
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (place pr2R green blue) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (place pr2L green blue) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green blue) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green blue) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (place pr2L green blue) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (place pr2R green blue) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green blue) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green blue) (grasp pr2R red) (place pr2R red table1)",
                        
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (place pr2R green blue) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (place pr2L green blue) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green blue) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green blue) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (place pr2L green blue) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (place pr2R green blue) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green blue) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green blue) (grasp pr2L red) (place pr2L red table1)"]
            numLoops=2*24*6
        elif getEnvInfo(nenv,"red") in [3]:
            solutions=[ "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green blue)"]
            numLoops=2*4*4
        elif getEnvInfo(nenv,"blue") in [2] and getEnvInfo(nenv,"red") in [0]: #bg rt
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue table1)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (place pr2R green blue)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (place pr2L green blue)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green blue) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (place pr2L green blue)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (place pr2R green blue)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green blue) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green blue)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"green") in [0,1] and getEnvInfo(nenv,"red") in [1,2,3]:
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
        elif getEnvInfo(nenv,"green") in [0,1] and getEnvInfo(nenv,"red") in [0]:
            solutions=["(grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2R green) (place pr2R green blue)"]
            numLoops=2*2*2
    
        elif getEnvInfo(nenv,"green") in [3] and getEnvInfo(nenv,"red") in [1,2,3]:
            solutions=["(grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L red) (place pr2L red table1)"]
            numLoops=2*2*2
        

    elif nset==13:
        goalString=["(on red table1) (held blue)", "(held blue) (on red table1)"]
        if getEnvInfo(nenv,"red") in [1,2,3]:
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue)"]
            numLoops=2*8*3
        elif getEnvInfo(nenv,"red") in [0]:
            solutions=["(grasp pr2R blue)","(grasp pr2L blue)"]
            numLoops=2*2*1

    elif nset==14:
        goalString=["(on red table1) (on blue table1)", "(on blue table1) (on red table1)"]
        if getEnvInfo(nenv,"red") in [0] and getEnvInfo(nenv,"blue") in [0]:
            solutions=[]
            numLoops=0
        elif getEnvInfo(nenv,"red") in [1,2,3] and getEnvInfo(nenv,"blue") in [1,2,3]:
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
        elif getEnvInfo(nenv,"red") in [1,2,3] and getEnvInfo(nenv,"blue") in [0]:
            solutions=["(grasp pr2L red) (place pr2L red table1)",
                    "(grasp pr2R red) (place pr2R red table1)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"red") in [0] and getEnvInfo(nenv,"blue") in [1,2,3]:
            solutions=["(grasp pr2L blue) (place pr2L blue table1)",
                    "(grasp pr2R blue) (place pr2R blue table1)"]
            numLoops=2*2*2

    elif nset==15:
        goalString=["(on red table1) (on blue red)", "(on blue red) (on red table1)"]
        if getEnvInfo(nenv,"red") in [0] and getEnvInfo(nenv,"blue") in [1]:
            solutions=[]
            numLoops=0

        elif getEnvInfo(nenv,"red") in [2] and getEnvInfo(nenv,"green") in [3] or getEnvInfo(nenv,"green") in [1] and getEnvInfo(nenv,"red") in [3]:
            solutions=["(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2R red table1) (place pr2L blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2L blue red) (place pr2R red table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue red)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue red)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2R blue red) (place pr2L red table1)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2L red table1) (place pr2R blue red)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2L red table1) (place pr2R blue red)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2R blue red) (place pr2L red table1)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue red)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue red)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2L blue red) (place pr2R red table1)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2R red table1) (place pr2L blue red)",
                        
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2R red table1) (place pr2L blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2L blue red) (place pr2R red table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue red)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue red)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2R blue red) (place pr2L red table1)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2L red table1) (place pr2R blue red)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2L red table1) (place pr2R blue red)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2R blue red) (place pr2L red table1)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue red)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue red)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2L blue red) (place pr2R red table1)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2R red table1) (place pr2L blue red)"]
            numLoops=2*24*6
        elif getEnvInfo(nenv,"green") in [1] and getEnvInfo(nenv,"red") in [0]:
            solutions=[ "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue red)"]
            numLoops=2*4*4

        elif getEnvInfo(nenv,"red") in [3]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red table1) (place pr2L blue red)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue red) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue red)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue red)", 
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue red) (place pr2L red table1)", 
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red table1) (place pr2R blue red)", 
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table1) (place pr2R blue red)", 
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue red) (place pr2L red table1)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue red)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue red)", 
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue red) (place pr2R red table1)", 
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table1) (place pr2L blue red)"]
            numLoops=2*12*4

        elif getEnvInfo(nenv,"red") in [1,2] and getEnvInfo(nenv,"blue") in [0,2,3]:
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
        elif getEnvInfo(nenv,"red") in [0] and getEnvInfo(nenv,"blue") in [0,2,3]:
            solutions=["(grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2R blue) (place pr2R blue red)"]
            numLoops=2*2*2
        
        elif getEnvInfo(nenv,"red") in [1,2] and getEnvInfo(nenv,"blue") in [1]:
            solutions=["(grasp pr2R red) (place pr2R red table1)",
                    "(grasp pr2L red) (place pr2L red table1)"]
            numLoops=2*2*2

    elif nset==16:
        goalString=["(on red table1) (on blue green)", "(on blue green) (on red table1)"]
        if getEnvInfo(nenv,"red") in [0] and getEnvInfo(nenv,"blue") in [2]:
            solutions=[]
            numLoops=0
        elif getEnvInfo(nenv,"green") in [1] and getEnvInfo(nenv,"red") in [3] or getEnvInfo(nenv,"green") in [3] and getEnvInfo(nenv,"red") in [1,2,3]: # grb
            solutions=["(grasp pr2R blue) (grasp pr2L green) (place pr2R blue green) (place pr2L green table1) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green table1) (place pr2R blue green) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2R green table1) (place pr2L blue green) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue green) (place pr2R green table1) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue green) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue green) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue green) (place pr2R green table1) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table1) (place pr2L blue green) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table1) (place pr2R blue green) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue green) (place pr2L green table1) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue green) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue green) (grasp pr2R red) (place pr2R red table1)",
                        
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue green) (place pr2L green table1) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green table1) (place pr2R blue green) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2R green table1) (place pr2L blue green) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue green) (place pr2R green table1) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue green) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue green) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue green) (place pr2R green table1) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table1) (place pr2L blue green) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table1) (place pr2R blue green) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue green) (place pr2L green table1) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue green) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue green) (grasp pr2L red) (place pr2L red table1)"]
            numLoops=2*24*6
        elif getEnvInfo(nenv,"red") in [2]:
            solutions=[ "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue green)"]
            numLoops=2*4*4
            
        elif getEnvInfo(nenv,"green") in [3] and getEnvInfo(nenv,"red") in [0]: #gb rt
            solutions=["(grasp pr2R blue) (grasp pr2L green) (place pr2R blue green) (place pr2L green table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green table1) (place pr2R blue green)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2R green table1) (place pr2L blue green)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue green) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue green) (place pr2R green table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table1) (place pr2L blue green)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table1) (place pr2R blue green)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue green) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue green)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"blue") in [0,1] and getEnvInfo(nenv,"red") in [1,2,3]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red table1) (place pr2L blue green)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue green) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue green) (place pr2L red table1)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red table1) (place pr2R blue green)",
                        "(grasp pr2R blue) (place pr2R blue green) (grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2R blue) (place pr2R blue green) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table1) (place pr2R blue green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue green) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue green) (place pr2R red table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table1) (place pr2L blue green)",
                        "(grasp pr2L blue) (place pr2L blue green) (grasp pr2L red) (place pr2L red table1)",
                        "(grasp pr2L blue) (place pr2L blue green) (grasp pr2R red) (place pr2R red table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"blue") in [0,1] and getEnvInfo(nenv,"red") in [0]:
            solutions=["(grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2R blue) (place pr2R blue green)"]
            numLoops=2*2*2
    
        elif getEnvInfo(nenv,"blue") in [2] and getEnvInfo(nenv,"red") in [1,2,3]:
            solutions=["(grasp pr2R red) (place pr2R red table1)",
                        "(grasp pr2L red) (place pr2L red table1)"]
            numLoops=2*2*2

    elif nset==17:
        goalString=["(on red green) (held green)", "(held green) (on red green)"]
        if getEnvInfo(nenv,"green") in [3] and getEnvInfo(nenv,"blue") in [1] or getEnvInfo(nenv,"blue") in [2]:
            solutions=["(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2R red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2L red green)",
                        
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2L red green)"]
            numLoops=2*8*5
        
        elif getEnvInfo(nenv,"green") in [1]:
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red green)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red green)"]
            numLoops=2*4*3
        
        elif getEnvInfo(nenv,"red") in [0,1,3]:
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red green)",
                        "(grasp pr2R red) (place pr2R red green) (grasp pr2R green)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red green)",
                        "(grasp pr2L red) (place pr2L red green) (grasp pr2R green)"]
            numLoops=2*6*3
        elif getEnvInfo(nenv,"red") in [2]:
            solutions=["(grasp pr2R green)","(grasp pr2L green)"]
            numLoops=2*2*1

    elif nset==18:
        goalString=["(on red green) (on green table1)", "(on green table1) (on red green)"]
        if getEnvInfo(nenv,"green") in [0] and getEnvInfo(nenv,"red") in [2]:
            solutions=[]
            numLoops=0
        elif getEnvInfo(nenv,"green") in [3] and getEnvInfo(nenv,"blue") in [1] or getEnvInfo(nenv,"blue") in [2] and getEnvInfo(nenv,"green") in [1]:
            solutions=["(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (place pr2L red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2L red green) (place pr2R green table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red green)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red green)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green table1)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red green)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (place pr2R red green)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2R red green) (place pr2L green table1)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red green)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red green)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green table1)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (place pr2L red green)",
                        
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (place pr2L red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2L red green) (place pr2R green table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red green)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red green)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green table1)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red green)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (place pr2R red green)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2R red green) (place pr2L green table1)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red green)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red green)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green table1)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (place pr2L red green)"]
            numLoops=2*24*6
        elif getEnvInfo(nenv,"blue") in [2]:
            solutions=[ "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red green)"]
            numLoops=2*4*4
            
        elif getEnvInfo(nenv,"green") in [1]:
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (place pr2L red green)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red green) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red green)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red green)", 
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green table1)", 
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red green)", 
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (place pr2R red green)", 
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red green) (place pr2L green table1)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red green)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red green)", 
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green table1)", 
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (place pr2L red green)"]
            numLoops=2*12*4

        elif getEnvInfo(nenv,"green") in [2,3] and getEnvInfo(nenv,"red") in [0,1,3]:
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (place pr2L red green)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red green) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red green)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red green)", 
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green table1)", 
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red green)", 
                        "(grasp pr2R red) (place pr2R red green) (grasp pr2R green) (place pr2R green table1)", 
                        "(grasp pr2R red) (place pr2R red green) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (place pr2R red green)", 
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red green) (place pr2L green table1)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red green)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red green)", 
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green table1)", 
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (place pr2L red green)", 
                        "(grasp pr2L red) (place pr2L red green) (grasp pr2R green) (place pr2R green table1)", 
                        "(grasp pr2L red) (place pr2L red green) (grasp pr2L green) (place pr2L green table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"green") in [0] and getEnvInfo(nenv,"red") in [0,1,3]:
            solutions=["(grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2R red) (place pr2R red green)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"green") in [2,3] and getEnvInfo(nenv,"red") in [2]:
            solutions=["(grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L green) (place pr2L green table1)"]
            numLoops=2*2*2

    elif nset==19:
        goalString=["(on red green) (on green blue)", "(on green blue) (on red green)"]
        if getEnvInfo(nenv,"green") in [3] and getEnvInfo(nenv,"red") in [2]:

            solutions=[]
            numLoops=0
        elif getEnvInfo(nenv,"blue") in [2] or getEnvInfo(nenv,"blue") in [1] and getEnvInfo(nenv,"green") in [0,1,2] and getEnvInfo(nenv,"red") in [0,1]:
            solutions=["(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green blue)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2L green blue) (place pr2R red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2R green blue) (place pr2L red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2L red green) (place pr2R green blue)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green blue) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green blue) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green blue)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2R green blue) (place pr2L red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2R red green) (place pr2L green blue)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2L green blue) (place pr2R red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green blue) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green blue) (grasp pr2L red) (place pr2L red green)",
                        
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green blue)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2L green blue) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2R green blue) (place pr2L red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2L red green) (place pr2R green blue)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green blue) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green blue) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green blue)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2R green blue) (place pr2L red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2R red green) (place pr2L green blue)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2L green blue) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green blue) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green blue) (grasp pr2L red) (place pr2L red green)"]
            
            numLoops=2*24*6

        elif getEnvInfo(nenv,"blue") in [0] and getEnvInfo(nenv,"green") in [0,1]:
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green blue)",
                    "(grasp pr2R red) (grasp pr2L green) (place pr2L green blue) (place pr2R red green)",
                    "(grasp pr2R green) (grasp pr2L red) (place pr2R green blue) (place pr2L red green)",
                    "(grasp pr2R green) (grasp pr2L red) (place pr2L red green) (place pr2R green blue)",
                    "(grasp pr2R green) (place pr2R green blue) (grasp pr2R red) (place pr2R red green)",
                    "(grasp pr2R green) (place pr2R green blue) (grasp pr2L red) (place pr2L red green)",
                    "(grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green blue)",
                    "(grasp pr2L red) (grasp pr2R green) (place pr2R green blue) (place pr2L red green)",
                    "(grasp pr2L green) (grasp pr2R red) (place pr2R red green) (place pr2L green blue)",
                    "(grasp pr2L green) (grasp pr2R red) (place pr2L green blue) (place pr2R red green)",
                    "(grasp pr2L green) (place pr2L green blue) (grasp pr2R red) (place pr2R red green)",
                    "(grasp pr2L green) (place pr2L green blue) (grasp pr2L red) (place pr2L red green)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"green") in [3] and getEnvInfo(nenv,"blue") in [1]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red green) (place pr2L blue table1)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (place pr2R red green)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (place pr2L red green)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red green) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red green) (place pr2R blue table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (place pr2L red green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (place pr2R red green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red green) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red green)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"red") in [2] and getEnvInfo(nenv,"blue") in [1]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue table1)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (place pr2R green blue)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (place pr2L green blue)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green blue) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (place pr2L green blue)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (place pr2R green blue)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green blue) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green blue)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"red") in [3]:
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green blue)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green blue) (place pr2R red green)",
                        "(grasp pr2R red) (place pr2R red green) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R red) (place pr2R red green) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green blue) (place pr2L red green)",
                        "(grasp pr2L red) (place pr2L red green) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L red) (place pr2L red green) (grasp pr2L green) (place pr2L green blue)"]
            numLoops=2*8*4

        elif getEnvInfo(nenv,"red") in [2]:
            solutions=["(grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L green) (place pr2L green blue)"]
            numLoops=2*2*2

        elif getEnvInfo(nenv,"green") in [3]:
            solutions=["(grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L red) (place pr2L red green)"]
            numLoops=2*2*2


    elif nset==20:
        #(held red) (on green blue)
        goalString=["(on red green) (held blue)", "(held blue) (on red green)"]
        
        if getEnvInfo(nenv,"green") in [1] and getEnvInfo(nenv,"blue") in [2]: # bgr
            solutions=[ "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2R green table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table1) (grasp pr2L red) (place pr2L red green)"]
            numLoops=2*4*5
        elif getEnvInfo(nenv,"blue") in [2]:
            solutions=[ "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue green)"]
            numLoops=2*4*3
        elif getEnvInfo(nenv,"green") in [0] or (getEnvInfo(nenv,"green") in [3] and getEnvInfo(nenv,"blue") in [0]): # gt or gbt 
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red green)",
                        "(grasp pr2R red) (place pr2R red green) (grasp pr2R blue)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red green)",
                        "(grasp pr2L red) (place pr2L red green) (grasp pr2R blue)"]
            numLoops=2*4*3
        
        elif getEnvInfo(nenv,"green") in [1] or (getEnvInfo(nenv,"green") in [3] and getEnvInfo(nenv,"blue") in [1]): # gr or gbr
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green table1) (grasp pr2R blue)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red green) (grasp pr2R blue)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (place pr2L red green) (grasp pr2R blue)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red green) (place pr2R green table1) (grasp pr2R blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red green) (grasp pr2R blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red green) (grasp pr2R blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green table1) (grasp pr2R blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (place pr2L red green) (grasp pr2R blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (place pr2R red green) (grasp pr2R blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red green) (place pr2L green table1) (grasp pr2R blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red green) (grasp pr2R blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red green) (grasp pr2R blue)",
                        
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green table1) (grasp pr2L blue)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red green) (grasp pr2L blue)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (place pr2L red green) (grasp pr2L blue)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red green) (place pr2R green table1) (grasp pr2L blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red green) (grasp pr2L blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red green) (grasp pr2L blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green table1) (grasp pr2L blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (place pr2L red green) (grasp pr2L blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (place pr2R red green) (grasp pr2L blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red green) (place pr2L green table1) (grasp pr2L blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red green) (grasp pr2L blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red green) (grasp pr2L blue)"]
            numLoops=2*24*5
        elif getEnvInfo(nenv,"red") in [2]:
            solutions=["(grasp pr2R blue)","(grasp pr2L blue)"]
            numLoops=2*2*1
        

    elif nset==21:
        #goalString=["(o.n r.ed g.reen) (o.n b.lue t.able1)", "(o.n b.lue t.able1) (on r.ed g.reen)"]
        goalString=["(on red green) (on blue table1)", "(on blue table1) (on red green)"]
        if getEnvInfo(nenv,"blue") in [0] and getEnvInfo(nenv,"red") in [2]:
            solutions=[]
            numLoops=0
        elif getEnvInfo(nenv,"green") in [3] and getEnvInfo(nenv,"blue") in [1] or getEnvInfo(nenv,"green") in [1] and getEnvInfo(nenv,"blue") in [1,2,3]: # gbr
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red green) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (place pr2L red green) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red green) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red green) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (place pr2L red green) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (place pr2R red green) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red green) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red green) (grasp pr2R blue) (place pr2R blue table1)",
                        
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red green) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (place pr2L red green) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red green) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red green) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (place pr2L red green) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (place pr2R red green) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red green) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red green) (grasp pr2L blue) (place pr2L blue table1)"]
            numLoops=2*24*6
        
        elif getEnvInfo(nenv,"blue") in [2]:
            solutions=[ "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red green)"]
            numLoops=2*4*4
        
        elif getEnvInfo(nenv,"green") in [1] and getEnvInfo(nenv,"blue") in [0]: #gr bt
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green table1)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red green)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (place pr2L red green)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red green) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (place pr2L red green)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (place pr2R red green)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red green) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red green)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"red") in [0,3] and getEnvInfo(nenv,"blue") in [1,2,3]:
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (place pr2L red green)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red green) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red green) (place pr2L blue table1)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (place pr2R red green)",
                        "(grasp pr2R red) (place pr2R red green) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R red) (place pr2R red green) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (place pr2R red green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red green) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red green) (place pr2R blue table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (place pr2L red green)",
                        "(grasp pr2L red) (place pr2L red green) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L red) (place pr2L red green) (grasp pr2R blue) (place pr2R blue table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"red") in [0,3] and getEnvInfo(nenv,"blue") in [0]:
            solutions=["(grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2R red) (place pr2R red green)"]
            numLoops=2*2*2
    
        elif getEnvInfo(nenv,"red") in [2] and getEnvInfo(nenv,"blue") in [1,2,3]:
            solutions=["(grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1)"]
            numLoops=2*2*2

    elif nset==22:
        ## r.ed <-g.reen 1 2
        # b.lue<-r.ed 3 1 
        # g.reen <- b.lue 2 3   
        goalString=["(on red green) (on blue red)", "(on blue red) (on red green)"]      
        if getEnvInfo(nenv,"red") in [2] and getEnvInfo(nenv,"blue") in [1]:

            solutions=[]
            numLoops=0
        elif getEnvInfo(nenv,"green") in [1] or getEnvInfo(nenv,"green") in [3] and getEnvInfo(nenv,"red") in [0,1,3] and getEnvInfo(nenv,"blue") in [0,3]:
            solutions=["(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2R blue red) (place pr2L red green)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2L red green) (place pr2R blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2R red green) (place pr2L blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2L blue red) (place pr2R red green)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red green) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red green) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2L blue red) (place pr2R red green)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2R red green) (place pr2L blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2R blue red) (place pr2L red green)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2L red green) (place pr2R blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red green) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red green) (grasp pr2L blue) (place pr2L blue red)",
                        
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2R blue red) (place pr2L red green)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2L red green) (place pr2R blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2R red green) (place pr2L blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2L blue red) (place pr2R red green)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red green) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red green) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2L blue red) (place pr2R red green)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2R red green) (place pr2L blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2R blue red) (place pr2L red green)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2L red green) (place pr2R blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red green) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red green) (grasp pr2L blue) (place pr2L blue red)"]
            
            numLoops=2*24*6

        elif getEnvInfo(nenv,"green") in [0] and getEnvInfo(nenv,"red") in [0,3]:
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2R blue red) (place pr2L red green)",
                    "(grasp pr2R blue) (grasp pr2L red) (place pr2L red green) (place pr2R blue red)",
                    "(grasp pr2R red) (grasp pr2L blue) (place pr2R red green) (place pr2L blue red)",
                    "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue red) (place pr2R red green)",
                    "(grasp pr2R red) (place pr2R red green) (grasp pr2R blue) (place pr2R blue red)",
                    "(grasp pr2R red) (place pr2R red green) (grasp pr2L blue) (place pr2L blue red)",
                    "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue red) (place pr2R red green)",
                    "(grasp pr2L blue) (grasp pr2R red) (place pr2R red green) (place pr2L blue red)",
                    "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue red) (place pr2L red green)",
                    "(grasp pr2L red) (grasp pr2R blue) (place pr2L red green) (place pr2R blue red)",
                    "(grasp pr2L red) (place pr2L red green) (grasp pr2R blue) (place pr2R blue red)",
                    "(grasp pr2L red) (place pr2L red green) (grasp pr2L blue) (place pr2L blue red)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"red") in [2] and getEnvInfo(nenv,"green") in [3]:
            solutions=["(grasp pr2R blue) (grasp pr2L green) (place pr2R blue red) (place pr2L green table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green table1) (place pr2R blue red)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2R green table1) (place pr2L blue red)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue red) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue red) (place pr2R green table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table1) (place pr2L blue red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table1) (place pr2R blue red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue red) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue red)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"blue") in [1] and getEnvInfo(nenv,"green") in [3]:
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red green) (place pr2L green table1)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red green)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (place pr2L red green)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red green) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red green) (place pr2R green table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (place pr2L red green)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (place pr2R red green)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red green) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red green)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"blue") in [2]:
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2R blue red) (place pr2L red green)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red green) (place pr2R blue red)",
                        "(grasp pr2R blue) (place pr2R blue red) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue red) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue red) (place pr2R red green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red green) (place pr2L blue red)",
                        "(grasp pr2L blue) (place pr2L blue red) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L blue) (place pr2L blue red) (grasp pr2L red) (place pr2L red green)"]
            numLoops=2*8*4

        elif getEnvInfo(nenv,"blue") in [1]:
            solutions=["(grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L red) (place pr2L red green)"]
            numLoops=2*2*2

        elif getEnvInfo(nenv,"red") in [2]:
            solutions=["(grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L blue) (place pr2L blue red)"]
            numLoops=2*2*2

    elif nset==23:
        goalString=["(on red blue) (held green)", "(held green) (on red blue)"]
        #goalString=["(on r.ed b.lue) (held g.reen)", "(held g.reen) (on r.ed b.lue)"]
        # red green 1 2
        if getEnvInfo(nenv,"green") in [3] and getEnvInfo(nenv,"blue") in [1]: # gbr
            solutions=[ "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (grasp pr2L red) (place pr2L red blue)"]
            numLoops=2*4*5
        elif getEnvInfo(nenv,"green") in [3]:
            solutions=[ "(grasp pr2R green) (grasp pr2L red) (place pr2R green blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green blue)"]
            numLoops=2*4*3
        elif getEnvInfo(nenv,"red") in [3]:
            solutions=["(grasp pr2R green)","(grasp pr2L green)"]
            numLoops=2*2*1
        elif getEnvInfo(nenv,"blue") in [0] or (getEnvInfo(nenv,"blue") in [2] and getEnvInfo(nenv,"green") in [0]): # bt or bgt 
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red blue)",
                        "(grasp pr2R red) (place pr2R red blue) (grasp pr2R green)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red blue)",
                        "(grasp pr2L red) (place pr2L red blue) (grasp pr2R green)"]
            numLoops=2*4*3
        
        elif getEnvInfo(nenv,"blue") in [1] or (getEnvInfo(nenv,"blue") in [2] and getEnvInfo(nenv,"green") in [1]): # br or bgr
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue table1) (grasp pr2R green)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (place pr2R red blue) (grasp pr2R green)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (place pr2L red blue) (grasp pr2R green)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red blue) (place pr2R blue table1) (grasp pr2R green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red blue) (grasp pr2R green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red blue) (grasp pr2R green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue table1) (grasp pr2R green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (place pr2L red blue) (grasp pr2R green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (place pr2R red blue) (grasp pr2R green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red blue) (place pr2L blue table1) (grasp pr2R green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red blue) (grasp pr2R green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red blue) (grasp pr2R green)",
                        
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue table1) (grasp pr2L green)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (place pr2R red blue) (grasp pr2L green)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (place pr2L red blue) (grasp pr2L green)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red blue) (place pr2R blue table1) (grasp pr2L green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red blue) (grasp pr2L green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red blue) (grasp pr2L green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue table1) (grasp pr2L green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (place pr2L red blue) (grasp pr2L green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (place pr2R red blue) (grasp pr2L green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red blue) (place pr2L blue table1) (grasp pr2L green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red blue) (grasp pr2L green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red blue) (grasp pr2L green)"]
            numLoops=2*24*5

    elif nset==24:
        goalString=["(on red blue) (on green table1)", "(on green table1) (on red blue)"]
        if getEnvInfo(nenv,"green") in [0] and getEnvInfo(nenv,"red") in [3]:
            solutions=[]
            numLoops=0
        elif getEnvInfo(nenv,"blue") in [2] and getEnvInfo(nenv,"green") in [1] or getEnvInfo(nenv,"blue") in [1] and getEnvInfo(nenv,"green") in [1,2,3]: # bgr
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (place pr2R red blue) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (place pr2L red blue) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red blue) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red blue) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (place pr2L red blue) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (place pr2R red blue) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red blue) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red blue) (grasp pr2R green) (place pr2R green table1)",
                        
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (place pr2R red blue) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (place pr2L red blue) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red blue) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red blue) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (place pr2L red blue) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (place pr2R red blue) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red blue) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red blue) (grasp pr2L green) (place pr2L green table1)"]
            numLoops=2*24*6
        elif getEnvInfo(nenv,"green") in [3]:
            solutions=[ "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red blue)"]
            numLoops=2*4*4
            
        elif getEnvInfo(nenv,"blue") in [1] and getEnvInfo(nenv,"green") in [0]: #br gt
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue table1)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (place pr2R red blue)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (place pr2L red blue)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red blue) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (place pr2L red blue)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (place pr2R red blue)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red blue) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red blue)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"red") in [0,2] and getEnvInfo(nenv,"green") in [1,2,3]:
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (place pr2L red blue)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red blue) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red blue) (place pr2L green table1)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red blue)",
                        "(grasp pr2R red) (place pr2R red blue) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2R red) (place pr2R red blue) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (place pr2R red blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red blue) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red blue) (place pr2R green table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (place pr2L red blue)",
                        "(grasp pr2L red) (place pr2L red blue) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L red) (place pr2L red blue) (grasp pr2R green) (place pr2R green table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"red") in [0,2] and getEnvInfo(nenv,"green") in [0]:
            solutions=["(grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2R red) (place pr2R red blue)"]
            numLoops=2*2*2
    
        elif getEnvInfo(nenv,"red") in [3] and getEnvInfo(nenv,"green") in [1,2,3]:
            solutions=["(grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L green) (place pr2L green table1)"]
            numLoops=2*2*2

    elif nset==25:
        #red -> green
        #green -> red
        goalString=["(on red blue) (on green red)", "(on green red) (on red blue)"]
        if getEnvInfo(nenv,"red") in [3] and getEnvInfo(nenv,"green") in [1]:

            solutions=[]
            numLoops=0
        elif getEnvInfo(nenv,"blue") in [1] or getEnvInfo(nenv,"blue") in [2] and getEnvInfo(nenv,"red") in [0,1,2] and getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2R green red) (place pr2L red blue)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2L red blue) (place pr2R green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2R red blue) (place pr2L green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2L green red) (place pr2R red blue)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red blue) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red blue) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2L green red) (place pr2R red blue)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2R red blue) (place pr2L green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2R green red) (place pr2L red blue)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2L red blue) (place pr2R green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red blue) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red blue) (grasp pr2L green) (place pr2L green red)",
                        
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2R green red) (place pr2L red blue)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (grasp pr2L red) (place pr2L red blue) (place pr2R green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2R red blue) (place pr2L green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (grasp pr2L green) (place pr2L green red) (place pr2R red blue)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red blue) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red blue) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2L green red) (place pr2R red blue)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (grasp pr2R red) (place pr2R red blue) (place pr2L green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2R green red) (place pr2L red blue)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (grasp pr2R green) (place pr2L red blue) (place pr2R green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red blue) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red blue) (grasp pr2L green) (place pr2L green red)"]
            
            numLoops=2*24*6

        elif getEnvInfo(nenv,"blue") in [0] and getEnvInfo(nenv,"red") in [0,2]:
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2R green red) (place pr2L red blue)",
                    "(grasp pr2R green) (grasp pr2L red) (place pr2L red blue) (place pr2R green red)",
                    "(grasp pr2R red) (grasp pr2L green) (place pr2R red blue) (place pr2L green red)",
                    "(grasp pr2R red) (grasp pr2L green) (place pr2L green red) (place pr2R red blue)",
                    "(grasp pr2R red) (place pr2R red blue) (grasp pr2R green) (place pr2R green red)",
                    "(grasp pr2R red) (place pr2R red blue) (grasp pr2L green) (place pr2L green red)",
                    "(grasp pr2L green) (grasp pr2R red) (place pr2L green red) (place pr2R red blue)",
                    "(grasp pr2L green) (grasp pr2R red) (place pr2R red blue) (place pr2L green red)",
                    "(grasp pr2L red) (grasp pr2R green) (place pr2R green red) (place pr2L red blue)",
                    "(grasp pr2L red) (grasp pr2R green) (place pr2L red blue) (place pr2R green red)",
                    "(grasp pr2L red) (place pr2L red blue) (grasp pr2R green) (place pr2R green red)",
                    "(grasp pr2L red) (place pr2L red blue) (grasp pr2L green) (place pr2L green red)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"red") in [3] and getEnvInfo(nenv,"blue") in [2]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green red) (place pr2L blue table1)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (place pr2R green red)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (place pr2L green red)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green red) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green red) (place pr2R blue table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (place pr2L green red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (place pr2R green red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green red) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green red)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"green") in [1] and getEnvInfo(nenv,"blue") in [2]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue table1)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (place pr2R red blue)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (place pr2L red blue)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red blue) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (place pr2L red blue)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (place pr2R red blue)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red blue) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red blue)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"green") in [3]:
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2R green red) (place pr2L red blue)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red blue) (place pr2R green red)",
                        "(grasp pr2R green) (place pr2R green red) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2R green) (place pr2R green red) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green red) (place pr2R red blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red blue) (place pr2L green red)",
                        "(grasp pr2L green) (place pr2L green red) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L green) (place pr2L green red) (grasp pr2L red) (place pr2L red blue)"]
            numLoops=2*8*4

        elif getEnvInfo(nenv,"green") in [1]:
            solutions=["(grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L red) (place pr2L red blue)"]
            numLoops=2*2*2

        elif getEnvInfo(nenv,"red") in [3]:
            solutions=["(grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L green) (place pr2L green red)"]
            numLoops=2*2*2

    elif nset==26:
        goalString=["(on red blue) (held blue)", "(held blue) (on red blue)"]
        if getEnvInfo(nenv,"blue") in [2] and getEnvInfo(nenv,"green") in [1] or getEnvInfo(nenv,"green") in [3]:
            solutions=["(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2R red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2L red blue)",
                        
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2R red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2L red blue)"]
            numLoops=2*8*5

        elif getEnvInfo(nenv,"blue") in [1]:
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red blue)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red blue)"]
            numLoops=2*4*3

        elif getEnvInfo(nenv,"red") in [0,1,3]:
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red blue)",
                        "(grasp pr2R red) (place pr2R red blue) (grasp pr2R blue)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red blue)",
                        "(grasp pr2L red) (place pr2L red blue) (grasp pr2R blue)"]
            numLoops=2*6*3
        elif getEnvInfo(nenv,"red") in [2]:
            solutions=["(grasp pr2R blue)","(grasp pr2L blue)"]
            numLoops=2*2*1

    elif nset==27:
        goalString=["(on red blue) (on blue table1)", "(on blue table1) (on red blue)"]
        if getEnvInfo(nenv,"blue") in [0] and getEnvInfo(nenv,"red") in [3]:
            solutions=[]
            numLoops=0

        elif getEnvInfo(nenv,"blue") in [2] and getEnvInfo(nenv,"green") in [1] or getEnvInfo(nenv,"green") in [3] and getEnvInfo(nenv,"blue") in [1]:
            solutions=["(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (place pr2L red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2L red blue) (place pr2R blue table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red blue)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red blue)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue table1)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (place pr2R red blue)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (place pr2R red blue)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2R red blue) (place pr2L blue table1)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red blue)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red blue)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue table1)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (place pr2L red blue)",
                        
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (place pr2L red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2L red blue) (place pr2R blue table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red blue)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red blue)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue table1)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (place pr2R red blue)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (place pr2R red blue)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2R red blue) (place pr2L blue table1)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red blue)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red blue)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue table1)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (place pr2L red blue)"]
            numLoops=2*24*6

        elif getEnvInfo(nenv,"blue") in [2]:
            solutions=[ "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red green)"]
            numLoops=2*4*4
        
        elif getEnvInfo(nenv,"blue") in [1]:
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (place pr2L red blue)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red blue) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red blue)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red blue)", 
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue table1)", 
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (place pr2R red blue)", 
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (place pr2R red blue)", 
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red blue) (place pr2L blue table1)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red blue)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red blue)", 
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue table1)", 
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (place pr2L red blue)"]
            numLoops=2*12*4

        elif getEnvInfo(nenv,"blue") in [2,3] and getEnvInfo(nenv,"red") in [0,1,2]:
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1) (place pr2L red blue)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red blue) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red) (place pr2R red blue)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red) (place pr2L red blue)", 
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue table1)", 
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1) (place pr2R red blue)", 
                        "(grasp pr2R red) (place pr2R red blue) (grasp pr2R blue) (place pr2R blue table1)", 
                        "(grasp pr2R red) (place pr2R red blue) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1) (place pr2R red blue)", 
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red blue) (place pr2L blue table1)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red) (place pr2R red blue)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red) (place pr2L red blue)", 
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue table1)", 
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1) (place pr2L red blue)", 
                        "(grasp pr2L red) (place pr2L red blue) (grasp pr2R blue) (place pr2R blue table1)", 
                        "(grasp pr2L red) (place pr2L red blue) (grasp pr2L blue) (place pr2L blue table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"blue") in [0] and getEnvInfo(nenv,"red") in [0,1,2]:
            solutions=["(grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2R red) (place pr2R red blue)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"blue") in [2,3] and getEnvInfo(nenv,"red") in [3]:
            solutions=["(grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1)"]
            numLoops=2*2*2


    elif nset==28:
        goalString=["(on red blue) (on blue green)", "(on blue green) (on red blue)"]
        if getEnvInfo(nenv,"blue") in [2] and getEnvInfo(nenv,"red") in [3]:

            solutions=[]
            numLoops=0
        elif getEnvInfo(nenv,"green") in [3] or getEnvInfo(nenv,"green") in [1] and getEnvInfo(nenv,"blue") in [0,1,3] and getEnvInfo(nenv,"red") in [0,1]:
            solutions=["(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue green)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2L blue green) (place pr2R red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2R blue green) (place pr2L red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2L red blue) (place pr2R blue green)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue green) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue green) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue green)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2R blue green) (place pr2L red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2R red blue) (place pr2L blue green)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2L blue green) (place pr2R red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue green) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue green) (grasp pr2L red) (place pr2L red blue)",
                        
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue green)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (grasp pr2L blue) (place pr2L blue green) (place pr2R red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2R blue green) (place pr2L red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (grasp pr2L red) (place pr2L red blue) (place pr2R blue green)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue green) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue green) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue green)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (grasp pr2R blue) (place pr2R blue green) (place pr2L red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2R red blue) (place pr2L blue green)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (grasp pr2R red) (place pr2L blue green) (place pr2R red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue green) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue green) (grasp pr2L red) (place pr2L red blue)"]
            
            numLoops=2*24*6

        elif getEnvInfo(nenv,"green") in [0] and getEnvInfo(nenv,"blue") in [0,1]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue green)",
                    "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue green) (place pr2R red blue)",
                    "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue green) (place pr2L red blue)",
                    "(grasp pr2R blue) (grasp pr2L red) (place pr2L red blue) (place pr2R blue green)",
                    "(grasp pr2R blue) (place pr2R blue green) (grasp pr2R red) (place pr2R red blue)",
                    "(grasp pr2R blue) (place pr2R blue green) (grasp pr2L red) (place pr2L red blue)",
                    "(grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue green)",
                    "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue green) (place pr2L red blue)",
                    "(grasp pr2L blue) (grasp pr2R red) (place pr2R red blue) (place pr2L blue green)",
                    "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue green) (place pr2R red blue)",
                    "(grasp pr2L blue) (place pr2L blue green) (grasp pr2R red) (place pr2R red blue)",
                    "(grasp pr2L blue) (place pr2L blue green) (grasp pr2L red) (place pr2L red blue)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"blue") in [2] and getEnvInfo(nenv,"green") in [1]:
            solutions=["(grasp pr2R red) (grasp pr2L green) (place pr2R red blue) (place pr2L green table1)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green table1) (place pr2R red blue)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green table1) (place pr2L red blue)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red blue) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L red) (place pr2L red blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red blue) (place pr2R green table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green table1) (place pr2L red blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green table1) (place pr2R red blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red blue) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L red) (place pr2L red blue)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"red") in [3] and getEnvInfo(nenv,"green") in [1]:
            solutions=["(grasp pr2R blue) (grasp pr2L green) (place pr2R blue green) (place pr2L green table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green table1) (place pr2R blue green)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2R green table1) (place pr2L blue green)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue green) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue green) (place pr2R green table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table1) (place pr2L blue green)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table1) (place pr2R blue green)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue green) (place pr2L green table1)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue green)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"red") in [2]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2R red blue) (place pr2L blue green)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue green) (place pr2R red blue)",
                        "(grasp pr2R red) (place pr2R red blue) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R red) (place pr2R red blue) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red blue) (place pr2R blue green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue green) (place pr2L red blue)",
                        "(grasp pr2L red) (place pr2L red blue) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L red) (place pr2L red blue) (grasp pr2L blue) (place pr2L blue green)"]
            numLoops=2*8*4

        elif getEnvInfo(nenv,"red") in [3]:
            solutions=["(grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L blue) (place pr2L blue green)"]
            numLoops=2*2*2

        elif getEnvInfo(nenv,"blue") in [2]:
            solutions=["(grasp pr2R red) (place pr2R red blue)",
                        "(grasp pr2L red) (place pr2L red blue)"]
            numLoops=2*2*2

    elif nset==29:
        goalString=["(held green) (held blue)", "(held blue) (held green)"]
        solutions=["(grasp pr2R blue) (grasp pr2L green)",
                    "(grasp pr2L blue) (grasp pr2R green)",
                    "(grasp pr2R green) (grasp pr2L blue)",
                    "(grasp pr2L green) (grasp pr2R blue)"]
        numLoops=2*4*2

    elif nset==30:
        goalString=["(held green) (on blue table1)", "(on blue table1) (held green)"]
        if getEnvInfo(nenv,"blue") in [1,2,3]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green)"]
            numLoops=2*8*3
        elif getEnvInfo(nenv,"blue") in [0]:
            solutions=["(grasp pr2R green)","(grasp pr2L green)"]
            numLoops=2*2*1

    elif nset==31:
        goalString=["(held green) (on blue red)", "(on blue red) (held green)"]
        if getEnvInfo(nenv,"green") in [1] and getEnvInfo(nenv,"red") in [3]: # grb
            solutions=[ "(grasp pr2L green) (grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue red)"]
            numLoops=2*4*5
        elif getEnvInfo(nenv,"green") in [1]:
            solutions=[ "(grasp pr2R green) (grasp pr2L blue) (place pr2R green red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green red)"]
            numLoops=2*4*3
        elif getEnvInfo(nenv,"red") in [0] or (getEnvInfo(nenv,"red") in [2] and getEnvInfo(nenv,"green") in [0]): #rbt or rgt 
            solutions=["(grasp pr2R blue) (grasp pr2L green) (place pr2R blue red)",
                        "(grasp pr2R blue) (place pr2R blue red) (grasp pr2R green)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue red)",
                        "(grasp pr2L blue) (place pr2L blue red) (grasp pr2R green)"]
            numLoops=2*4*3
        
        elif getEnvInfo(nenv,"red") in [3] or (getEnvInfo(nenv,"red") in [2] and getEnvInfo(nenv,"green") in [3]): # rb or rgb
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2R blue red) (place pr2L red table1) (grasp pr2R green)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red table1) (place pr2R blue red) (grasp pr2R green)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red table1) (place pr2L blue red) (grasp pr2R green)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue red) (place pr2R red table1) (grasp pr2R green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue red) (grasp pr2R green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue red) (grasp pr2R green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue red) (place pr2R red table1) (grasp pr2R green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table1) (place pr2L blue red) (grasp pr2R green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table1) (place pr2R blue red) (grasp pr2R green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue red) (place pr2L red table1) (grasp pr2R green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue red) (grasp pr2R green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue red) (grasp pr2R green)",
                        
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue red) (place pr2L red table1) (grasp pr2L green)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red table1) (place pr2R blue red) (grasp pr2L green)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red table1) (place pr2L blue red) (grasp pr2L green)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue red) (place pr2R red table1) (grasp pr2L green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue red) (grasp pr2L green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue red) (grasp pr2L green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue red) (place pr2R red table1) (grasp pr2L green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table1) (place pr2L blue red) (grasp pr2L green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table1) (place pr2R blue red) (grasp pr2L green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue red) (place pr2L red table1) (grasp pr2L green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue red) (grasp pr2L green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue red) (grasp pr2L green)"]
            numLoops=2*24*5
        elif getEnvInfo(nenv,"blue") in [1]:
            solutions=["(grasp pr2R green)","(grasp pr2L green)"]
            numLoops=2*2*1

    elif nset==32:
        goalString=["(on green table1) (held blue)", "(held blue) (on green table1)"]
        if getEnvInfo(nenv,"blue") in [1,2,3]:
            solutions=["(grasp pr2R red) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L red)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L red)"]
            numLoops=2*8*3
        elif getEnvInfo(nenv,"blue") in [0]:
            solutions=["(grasp pr2R blue)","(grasp pr2L blue)"]
            numLoops=2*2*1

    elif nset==33:
        goalString=["(on green table1) (on blue table1)", "(on blue table1) (on green table1)"]
        if getEnvInfo(nenv,"blue") in [0] and getEnvInfo(nenv,"green") in [0]:
            solutions=[]
            numLoops=0
        elif getEnvInfo(nenv,"blue") in [1,2,3] and getEnvInfo(nenv,"green") in [1,2,3]:
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
        elif getEnvInfo(nenv,"blue") in [1,2,3] and getEnvInfo(nenv,"green") in [0]:
            solutions=["(grasp pr2L blue) (place pr2L blue table1)",
                    "(grasp pr2R blue) (place pr2R blue table1)"]
            numLoops=2*2*2
        elif getEnvInfo(nenv,"blue") in [0] and getEnvInfo(nenv,"green") in [1,2,3]:
            solutions=["(grasp pr2L green) (place pr2L green table1)",
                    "(grasp pr2R green) (place pr2R green table1)"]
            numLoops=2*2*2

    elif nset==34:
        goalString=["(on green table1) (on blue red)", "(on blue red) (on green table1)"]
        if getEnvInfo(nenv,"green") in [0] and getEnvInfo(nenv,"blue") in [1]:
            solutions=[]
            numLoops=0
        elif getEnvInfo(nenv,"red") in [2] and getEnvInfo(nenv,"green") in [3] or getEnvInfo(nenv,"red") in [3] and getEnvInfo(nenv,"green") in [1,2,3]: # rgb
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2R blue red) (place pr2L red table1) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red table1) (place pr2R blue red) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red table1) (place pr2L blue red) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue red) (place pr2R red table1) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue red) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue red) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue red) (place pr2R red table1) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table1) (place pr2L blue red) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table1) (place pr2R blue red) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue red) (place pr2L red table1) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue red) (grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue red) (grasp pr2R green) (place pr2R green table1)",
                        
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2R blue red) (place pr2L red table1) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red table1) (place pr2R blue red) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red table1) (place pr2L blue red) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue red) (place pr2R red table1) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue red) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue red) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue red) (place pr2R red table1) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table1) (place pr2L blue red) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table1) (place pr2R blue red) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue red) (place pr2L red table1) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue red) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue red) (grasp pr2L green) (place pr2L green table1)"]
            numLoops=2*24*6
        elif getEnvInfo(nenv,"green") in [1]:
            solutions=[ "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue red)"]
            numLoops=2*4*4
        elif getEnvInfo(nenv,"red") in [3] and getEnvInfo(nenv,"green") in [0]: #rb gt
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2R blue red) (place pr2L red table1)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red table1) (place pr2R blue red)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red table1) (place pr2L blue red)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue red) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue red) (place pr2R red table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table1) (place pr2L blue red)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table1) (place pr2R blue red)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue red) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue red)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"blue") in [0,2] and getEnvInfo(nenv,"green") in [1,2,3]:
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
                        "(grasp pr2L blue) (place pr2L blue red) (grasp pr2L green) (place pr2L green table1)",
                        "(grasp pr2L blue) (place pr2L blue red) (grasp pr2R green) (place pr2R green table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"blue") in [0,2] and getEnvInfo(nenv,"green") in [0]:
            solutions=["(grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2R blue) (place pr2R blue red)"]
            numLoops=2*2*2
    
        elif getEnvInfo(nenv,"blue") in [1] and getEnvInfo(nenv,"green") in [1,2,3]:
            solutions=["(grasp pr2R green) (place pr2R green table1)",
                        "(grasp pr2L green) (place pr2L green table1)"]
            numLoops=2*2*2

    elif nset==35:
        goalString=["(on green table1) (on blue green)", "(on blue green) (on green table1)"]
        if getEnvInfo(nenv,"green") in [0] and getEnvInfo(nenv,"blue") in [2]:
            solutions=[]
            numLoops=0
        elif getEnvInfo(nenv,"green") in [1] and getEnvInfo(nenv,"red") in [3] or getEnvInfo(nenv,"red") in [2] and getEnvInfo(nenv,"green") in [3]:
            solutions=["(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2R green table1) (place pr2L blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2L blue green) (place pr2R green table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue green)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue green)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2R blue green) (place pr2L green table1)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2L green table1) (place pr2R blue green)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2L green table1) (place pr2R blue green)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2R blue green) (place pr2L green table1)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue green)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue green)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2L blue green) (place pr2R green table1)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2R green table1) (place pr2L blue green)",
                        
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2R green table1) (place pr2L blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2L blue green) (place pr2R green table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue green)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue green)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2R blue green) (place pr2L green table1)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2L green table1) (place pr2R blue green)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2L green table1) (place pr2R blue green)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2R blue green) (place pr2L green table1)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue green)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue green)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2L blue green) (place pr2R green table1)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2R green table1) (place pr2L blue green)"]
            numLoops=2*24*6
        elif getEnvInfo(nenv,"red") in [2]:
            solutions=[ "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue green)"]
            numLoops=2*4*4

        elif getEnvInfo(nenv,"green") in [3]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green table1) (place pr2L blue green)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue green) (place pr2R green table1)",
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2R blue) (place pr2R blue green)", 
                        "(grasp pr2R green) (place pr2R green table1) (grasp pr2L blue) (place pr2L blue green)", 
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue green) (place pr2L green table1)", 
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green table1) (place pr2R blue green)", 
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green table1) (place pr2R blue green)", 
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue green) (place pr2L green table1)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2R blue) (place pr2R blue green)", 
                        "(grasp pr2L green) (place pr2L green table1) (grasp pr2L blue) (place pr2L blue green)", 
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue green) (place pr2R green table1)", 
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green table1) (place pr2L blue green)"]
            numLoops=2*12*4

        elif getEnvInfo(nenv,"green") in [1,2] and getEnvInfo(nenv,"blue") in [0,1,3]:
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
        elif getEnvInfo(nenv,"green") in [0] and getEnvInfo(nenv,"blue") in [0,1,3]:
            solutions=["(grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2R blue) (place pr2R blue green)"]
            numLoops=2*2*2
        
        elif getEnvInfo(nenv,"green") in [1,2] and getEnvInfo(nenv,"blue") in [2]:
            solutions=["(grasp pr2R green) (place pr2R green table1)",
                    "(grasp pr2L green) (place pr2L green table1)"]
            numLoops=2*2*2

    elif nset==36:
        goalString=["(on green red) (held blue)", "(held blue) (on green red)"]
        if getEnvInfo(nenv,"blue") in [1] and getEnvInfo(nenv,"red") in [2]: # brg
            solutions=[ "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table1) (grasp pr2L green) (place pr2L green red)"]
            numLoops=2*4*5
        elif getEnvInfo(nenv,"blue") in [1]:
            solutions=[ "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue red)"]
            numLoops=2*4*3

        elif getEnvInfo(nenv,"red") in [0] or (getEnvInfo(nenv,"red") in [3] and getEnvInfo(nenv,"blue") in [0]): # rt or rbt 
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green red)",
                        "(grasp pr2R green) (place pr2R green red) (grasp pr2R blue)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green red)",
                        "(grasp pr2L green) (place pr2L green red) (grasp pr2R blue)"]
            numLoops=2*4*3
        
        elif getEnvInfo(nenv,"red") in [2] or (getEnvInfo(nenv,"red") in [3] and getEnvInfo(nenv,"blue") in [2]): # rg or rbg
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2R green red) (place pr2L red table1) (grasp pr2R blue)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red table1) (place pr2R green red) (grasp pr2R blue)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red table1) (place pr2L green red) (grasp pr2R blue)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green red) (place pr2R red table1) (grasp pr2R blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green red) (grasp pr2R blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green red) (grasp pr2R blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green red) (place pr2R red table1) (grasp pr2R blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table1) (place pr2L green red) (grasp pr2R blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table1) (place pr2R green red) (grasp pr2R blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green red) (place pr2L red table1) (grasp pr2R blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green red) (grasp pr2R blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green red) (grasp pr2R blue)",
                        
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green red) (place pr2L red table1) (grasp pr2L blue)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red table1) (place pr2R green red) (grasp pr2L blue)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red table1) (place pr2L green red) (grasp pr2L blue)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green red) (place pr2R red table1) (grasp pr2L blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green red) (grasp pr2L blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green red) (grasp pr2L blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green red) (place pr2R red table1) (grasp pr2L blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table1) (place pr2L green red) (grasp pr2L blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table1) (place pr2R green red) (grasp pr2L blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green red) (place pr2L red table1) (grasp pr2L blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green red) (grasp pr2L blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green red) (grasp pr2L blue)"]
            numLoops=2*24*5
        elif getEnvInfo(nenv,"green") in [1]:
            solutions=["(grasp pr2R blue)","(grasp pr2L blue)"]
            numLoops=2*2*1

    elif nset==37:
        goalString=["(on green red) (on blue table1)", "(on blue table1) (on green red)"]
        if getEnvInfo(nenv,"blue") in [0] and getEnvInfo(nenv,"green") in [2]:
            solutions=[]
            numLoops=0
        elif getEnvInfo(nenv,"red") in [3] and getEnvInfo(nenv,"blue") in [2] or getEnvInfo(nenv,"red") in [2] and getEnvInfo(nenv,"blue") in [1,2,3]: # rbg
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2R green red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red table1) (place pr2R green red) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red table1) (place pr2L green red) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green red) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green red) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table1) (place pr2L green red) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table1) (place pr2R green red) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green red) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green red) (grasp pr2R blue) (place pr2R blue table1)",
                        
                        "(grasp pr2R green) (grasp pr2L red) (place pr2R green red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red table1) (place pr2R green red) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red table1) (place pr2L green red) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green red) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green red) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table1) (place pr2L green red) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table1) (place pr2R green red) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green red) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green red) (grasp pr2L blue) (place pr2L blue table1)"]
            numLoops=2*24*6

        elif getEnvInfo(nenv,"blue") in [1]:
            solutions=[ "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green red)"]
            numLoops=2*4*4

        elif getEnvInfo(nenv,"red") in [2] and getEnvInfo(nenv,"blue") in [0]: #gr bt
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2R green red) (place pr2L red table1)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red table1) (place pr2R green red)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red table1) (place pr2L green red)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green red) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green red) (place pr2R red table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table1) (place pr2L green red)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table1) (place pr2R green red)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green red) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green red)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"green") in [0,3] and getEnvInfo(nenv,"blue") in [1,2,3]:
            solutions=["(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (place pr2L green red)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green red) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2R green red) (place pr2L blue table1)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (place pr2R green red)",
                        "(grasp pr2R green) (place pr2R green red) (grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2R green) (place pr2R green red) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (place pr2R green red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green red) (place pr2L blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green red) (place pr2R blue table1)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (place pr2L green red)",
                        "(grasp pr2L green) (place pr2L green red) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L green) (place pr2L green red) (grasp pr2R blue) (place pr2R blue table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"green") in [0,3] and getEnvInfo(nenv,"blue") in [0]:
            solutions=["(grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2R green) (place pr2R green red)"]
            numLoops=2*2*2
    
        elif getEnvInfo(nenv,"green") in [1] and getEnvInfo(nenv,"blue") in [1,2,3]:
            solutions=["(grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1)"]
            numLoops=2*2*2

    elif nset==38:
        goalString=["(on green red) (on blue green)", "(on blue green) (on green red)"]
        # b r
        if getEnvInfo(nenv,"green") in [1] and getEnvInfo(nenv,"blue") in [2]:

            solutions=[]
            numLoops=0
        elif getEnvInfo(nenv,"red") in [2] or getEnvInfo(nenv,"red") in [3] and getEnvInfo(nenv,"green") in [0,2,3] and getEnvInfo(nenv,"blue") in [0,3]:
            solutions=["(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2R blue green) (place pr2L green red)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2L green red) (place pr2R blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2R green red) (place pr2L blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2L blue green) (place pr2R green red)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green red) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green red) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2L blue green) (place pr2R green red)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2R green red) (place pr2L blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2R blue green) (place pr2L green red)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2L green red) (place pr2R blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green red) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green red) (grasp pr2L blue) (place pr2L blue green)",
                        
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2R blue green) (place pr2L green red)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2L green red) (place pr2R blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2R green red) (place pr2L blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2L blue green) (place pr2R green red)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green red) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green red) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2L blue green) (place pr2R green red)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2R green red) (place pr2L blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2R blue green) (place pr2L green red)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2L green red) (place pr2R blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green red) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green red) (grasp pr2L blue) (place pr2L blue green)"]
            
            numLoops=2*24*6

        elif getEnvInfo(nenv,"red") in [0] and getEnvInfo(nenv,"green") in [0,3]:
            solutions=["(grasp pr2R blue) (grasp pr2L green) (place pr2R blue green) (place pr2L green red)",
                    "(grasp pr2R blue) (grasp pr2L green) (place pr2L green red) (place pr2R blue green)",
                    "(grasp pr2R green) (grasp pr2L blue) (place pr2R green red) (place pr2L blue green)",
                    "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue green) (place pr2R green red)",
                    "(grasp pr2R green) (place pr2R green red) (grasp pr2R blue) (place pr2R blue green)",
                    "(grasp pr2R green) (place pr2R green red) (grasp pr2L blue) (place pr2L blue green)",
                    "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue green) (place pr2R green red)",
                    "(grasp pr2L blue) (grasp pr2R green) (place pr2R green red) (place pr2L blue green)",
                    "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue green) (place pr2L green red)",
                    "(grasp pr2L green) (grasp pr2R blue) (place pr2L green red) (place pr2R blue green)",
                    "(grasp pr2L green) (place pr2L green red) (grasp pr2R blue) (place pr2R blue green)",
                    "(grasp pr2L green) (place pr2L green red) (grasp pr2L blue) (place pr2L blue green)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"green") in [1] and getEnvInfo(nenv,"red") in [3]:
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2R blue green) (place pr2L red table1)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red table1) (place pr2R blue green)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red table1) (place pr2L blue green)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue green) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue green) (place pr2R red table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table1) (place pr2L blue green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table1) (place pr2R blue green)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue green) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue green)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"blue") in [2] and getEnvInfo(nenv,"red") in [3]:
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2R green red) (place pr2L red table1)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red table1) (place pr2R green red)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red table1) (place pr2L green red)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green red) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green red) (place pr2R red table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table1) (place pr2L green red)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table1) (place pr2R green red)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green red) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green red)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"blue") in [1]:
            solutions=["(grasp pr2R blue) (grasp pr2L green) (place pr2R blue green) (place pr2L green red)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green red) (place pr2R blue green)",
                        "(grasp pr2R blue) (place pr2R blue green) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2R blue) (place pr2R blue green) (grasp pr2L green) (place pr2L green red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue green) (place pr2R green red)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green red) (place pr2L blue green)",
                        "(grasp pr2L blue) (place pr2L blue green) (grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L blue) (place pr2L blue green) (grasp pr2L green) (place pr2L green red)"]
            numLoops=2*8*4

        elif getEnvInfo(nenv,"blue") in [2]:
            solutions=["(grasp pr2R green) (place pr2R green red)",
                        "(grasp pr2L green) (place pr2L green red)"]
            numLoops=2*2*2

        elif getEnvInfo(nenv,"green") in [1]:
            solutions=["(grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L blue) (place pr2L blue green)"]
            numLoops=2*2*2

    elif nset==39:
        # 1 3
        goalString=["(on green blue) (held blue)", "(held blue) (on green blue)"]
        if getEnvInfo(nenv,"blue") in [2]:
            solutions=["(grasp pr2R blue) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green blue)"]
            numLoops=2*2*3

        elif getEnvInfo(nenv,"blue") in [1] and getEnvInfo(nenv,"red") in [2] or getEnvInfo(nenv,"red") in [3]:
            solutions=["(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2R green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2L green blue)",
                        
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2R green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2L green blue)"]
            numLoops=2*8*5

        elif getEnvInfo(nenv,"blue") in [2]:
            solutions=["(grasp pr2R blue) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2R green blue)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green blue)"]
            numLoops=2*4*3

        elif getEnvInfo(nenv,"green") in [0,1,2]:
            solutions=["(grasp pr2R blue) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2R green blue)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green blue)"]
            numLoops=2*6*3
        elif getEnvInfo(nenv,"green") in [3]:
            solutions=["(grasp pr2R blue)","(grasp pr2L blue)"]
            numLoops=2*2*1

    elif nset==40:
        goalString=["(on green blue) (on blue table1)", "(on blue table1) (on green blue)"]
        # 1 3
        if getEnvInfo(nenv,"blue") in [0] and getEnvInfo(nenv,"green") in [3]:
            solutions=[]
            numLoops=0

        elif getEnvInfo(nenv,"blue") in [1] and getEnvInfo(nenv,"red") in [2] or getEnvInfo(nenv,"red") in [3] and getEnvInfo(nenv,"blue") in [2]:
            solutions=["(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (place pr2L green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2L green blue) (place pr2R blue table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green blue)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green blue)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue table1)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (place pr2R green blue)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (place pr2R green blue)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2R green blue) (place pr2L blue table1)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green blue)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green blue)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue table1)", 
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (place pr2L green blue)",
                        
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (place pr2L green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2L green blue) (place pr2R blue table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green blue)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green blue)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue table1)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (place pr2R green blue)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (place pr2R green blue)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2R green blue) (place pr2L blue table1)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green blue)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green blue)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue table1)", 
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (place pr2L green blue)"]
            numLoops=2*24*6
        elif getEnvInfo(nenv,"red") in [3]:
            solutions=[ "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green blue)"]
            numLoops=2*4*4
        
        elif getEnvInfo(nenv,"blue") in [2]:
            solutions=["(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (place pr2L green blue)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green blue) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green blue)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green blue)", 
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue table1)", 
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (place pr2R green blue)", 
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (place pr2R green blue)", 
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green blue) (place pr2L blue table1)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green blue)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green blue)", 
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue table1)", 
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (place pr2L green blue)"]
            numLoops=2*12*4

        elif getEnvInfo(nenv,"blue") in [1,3] and getEnvInfo(nenv,"green") in [0,1,2]:
            solutions=["(grasp pr2R blue) (grasp pr2L green) (place pr2R blue table1) (place pr2L green blue)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2L green blue) (place pr2R blue table1)",
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2R green) (place pr2R green blue)", 
                        "(grasp pr2R blue) (place pr2R blue table1) (grasp pr2L green) (place pr2L green blue)", 
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue table1)", 
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue table1) (place pr2R green blue)", 
                        "(grasp pr2R green) (place pr2R green blue) (grasp pr2R blue) (place pr2R blue table1)", 
                        "(grasp pr2R green) (place pr2R green blue) (grasp pr2L blue) (place pr2L blue table1)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue table1) (place pr2R green blue)", 
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2R green blue) (place pr2L blue table1)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2R green) (place pr2R green blue)", 
                        "(grasp pr2L blue) (place pr2L blue table1) (grasp pr2L green) (place pr2L green blue)", 
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue table1)", 
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue table1) (place pr2L green blue)", 
                        "(grasp pr2L green) (place pr2L green blue) (grasp pr2R blue) (place pr2R blue table1)", 
                        "(grasp pr2L green) (place pr2L green blue) (grasp pr2L blue) (place pr2L blue table1)"]
            numLoops=2*16*4
        elif getEnvInfo(nenv,"blue") in [0] and getEnvInfo(nenv,"green") in [0,1,2]:
            solutions=["(grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2R green) (place pr2R green blue)"]
            numLoops=2*2*2

        elif getEnvInfo(nenv,"blue") in [1,3] and getEnvInfo(nenv,"green") in [3]:
            solutions=["(grasp pr2R blue) (place pr2R blue table1)",
                        "(grasp pr2L blue) (place pr2L blue table1)"]
            numLoops=2*2*2

    elif nset==41:
        goalString=["(on green blue) (on blue red)", "(on blue red) (on green blue)"]
        # 1 2
        if getEnvInfo(nenv,"blue") in [1] and getEnvInfo(nenv,"green") in [3]:

            solutions=[]
            numLoops=0
        elif getEnvInfo(nenv,"red") in [3] or getEnvInfo(nenv,"red") in [2] and getEnvInfo(nenv,"blue") in [0,2,3] and getEnvInfo(nenv,"green") in [0,2]:
            solutions=["(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue red)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2L blue red) (place pr2R green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2R blue red) (place pr2L green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2L green blue) (place pr2R blue red)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue red) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue red) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue red)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2R blue red) (place pr2L green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2R green blue) (place pr2L blue red)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2L blue red) (place pr2R green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue red) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue red) (grasp pr2L green) (place pr2L green blue)",
                        
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue red)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2L blue red) (place pr2R green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2R blue red) (place pr2L green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2L green blue) (place pr2R blue red)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue red) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue red) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue red)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2R blue red) (place pr2L green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2R green blue) (place pr2L blue red)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2L blue red) (place pr2R green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue red) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue red) (grasp pr2L green) (place pr2L green blue)"]
            
            numLoops=2*24*6

        elif getEnvInfo(nenv,"red") in [0] and getEnvInfo(nenv,"blue") in [0,2]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue red)",
                    "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue red) (place pr2R green blue)",
                    "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue red) (place pr2L green blue)",
                    "(grasp pr2R blue) (grasp pr2L green) (place pr2L green blue) (place pr2R blue red)",
                    "(grasp pr2R blue) (place pr2R blue red) (grasp pr2R green) (place pr2R green blue)",
                    "(grasp pr2R blue) (place pr2R blue red) (grasp pr2L green) (place pr2L green blue)",
                    "(grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue red)",
                    "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue red) (place pr2L green blue)",
                    "(grasp pr2L blue) (grasp pr2R green) (place pr2R green blue) (place pr2L blue red)",
                    "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue red) (place pr2R green blue)",
                    "(grasp pr2L blue) (place pr2L blue red) (grasp pr2R green) (place pr2R green blue)",
                    "(grasp pr2L blue) (place pr2L blue red) (grasp pr2L green) (place pr2L green blue)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"blue") in [1] and getEnvInfo(nenv,"red") in [2]:
            solutions=["(grasp pr2R green) (grasp pr2L red) (place pr2R green blue) (place pr2L red table1)",
                        "(grasp pr2R green) (grasp pr2L red) (place pr2L red table1) (place pr2R green blue)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2R red table1) (place pr2L green blue)",
                        "(grasp pr2R red) (grasp pr2L green) (place pr2L green blue) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (place pr2L green blue)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2L green blue) (place pr2R red table1)",
                        "(grasp pr2L green) (grasp pr2R red) (place pr2R red table1) (place pr2L green blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2L red table1) (place pr2R green blue)",
                        "(grasp pr2L red) (grasp pr2R green) (place pr2R green blue) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (place pr2L green blue)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"green") in [3] and getEnvInfo(nenv,"red") in [2]:
            solutions=["(grasp pr2R blue) (grasp pr2L red) (place pr2R blue red) (place pr2L red table1)",
                        "(grasp pr2R blue) (grasp pr2L red) (place pr2L red table1) (place pr2R blue red)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2R red table1) (place pr2L blue red)",
                        "(grasp pr2R red) (grasp pr2L blue) (place pr2L blue red) (place pr2R red table1)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2L blue red) (place pr2R red table1)",
                        "(grasp pr2L blue) (grasp pr2R red) (place pr2R red table1) (place pr2L blue red)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2L red table1) (place pr2R blue red)",
                        "(grasp pr2L red) (grasp pr2R blue) (place pr2R blue red) (place pr2L red table1)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (place pr2L blue red)"]
            numLoops=2*12*4
        
        elif getEnvInfo(nenv,"green") in [1]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2R green blue) (place pr2L blue red)",
                        "(grasp pr2R green) (grasp pr2L blue) (place pr2L blue red) (place pr2R green blue)",
                        "(grasp pr2R green) (place pr2R green blue) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2R green) (place pr2R green blue) (grasp pr2L blue) (place pr2L blue red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2L green blue) (place pr2R blue red)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue red) (place pr2L green blue)",
                        "(grasp pr2L green) (place pr2L green blue) (grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L green) (place pr2L green blue) (grasp pr2L blue) (place pr2L blue red)"]
            numLoops=2*8*4

        elif getEnvInfo(nenv,"green") in [3]:
            solutions=["(grasp pr2R blue) (place pr2R blue red)",
                        "(grasp pr2L blue) (place pr2L blue red)"]
            numLoops=2*2*2

        elif getEnvInfo(nenv,"blue") in [1]:
            solutions=["(grasp pr2R green) (place pr2R green blue)",
                        "(grasp pr2L green) (place pr2L green blue)"]
            numLoops=2*2*2

    elif nset==42:
        goalString=["(on blue green) (held green)", "(held green) (on blue green)"]
        if getEnvInfo(nenv,"green") in [1]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue green)"]
            numLoops=2*2*3
        
        elif getEnvInfo(nenv,"green") in [1] and getEnvInfo(nenv,"red") in [3] or getEnvInfo(nenv,"red") in [2]:
            solutions=["(grasp pr2L red) (place pr2L red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2R blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L red) (place pr2L red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2L blue green)",
                        
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R green) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2R blue) (grasp pr2L green) (place pr2R blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L green) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2R red) (place pr2R red table1) (grasp pr2L blue) (grasp pr2R green) (place pr2L blue green)"]
            numLoops=2*8*5
        
        elif getEnvInfo(nenv,"green") in [3]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue green)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue green)"]
            numLoops=2*4*3
        
        elif getEnvInfo(nenv,"blue") in [0,1,3]:
            solutions=["(grasp pr2R green) (grasp pr2L blue) (place pr2L blue green)",
                        "(grasp pr2R blue) (grasp pr2L green) (place pr2R blue green)",
                        "(grasp pr2R blue) (place pr2R blue green) (grasp pr2R green)",
                        "(grasp pr2L green) (grasp pr2R blue) (place pr2R blue green)",
                        "(grasp pr2L blue) (grasp pr2R green) (place pr2L blue green)",
                        "(grasp pr2L blue) (place pr2L blue green) (grasp pr2R green)"]
            numLoops=2*6*3
        elif getEnvInfo(nenv,"blue") in [2]:
            solutions=["(grasp pr2R green)","(grasp pr2L green)"]
            numLoops=2*2*1




    else:
        print("\nNot implemented: set "+str(nset).zfill(3))

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
    
    if nenv >200:
        solutions, _, numLoops = getData(nset=nset, nenv=nenv-200)
        print(solutions, numLoops)
        print(getEnvInfo(nenv-200,"red"), getEnvInfo(nenv-200,"green"), getEnvInfo(nenv-200,"blue"))
    else:
        print("Not implemented nenv: "+str(nenv)+" nset: "+str(nset)+"\n\n")
        
        #num=1
        #for i in range(0,len(Sets),2):
        #        print("elif nset=="+str(num)+":")
        #        print('\tgoalString=["'+Sets[i]+'", "'+Sets[i+1]+'"]')
        #        print()
        #        num+=1

        for nenv in range(201,253):
            strNoSol=str(nenv).zfill(3)+"\t"
            for nset in range(1,13):
                _, _, numLoops = getData1(nset=nset, nenv=nenv-200)
                if numLoops==0:
                    strNoSol=strNoSol+" "+ str(nset).zfill(2)
            
            for nset in range(1,43):
                _, _, numLoops = getData(nset=nset, nenv=nenv-200)
                if numLoops==0:
                    strNoSol=strNoSol+" "+ str(nset).zfill(3)

            print(strNoSol)


        #for i in range(len(test)):
        #    for j in range(i,len(test)):
        #        if not i==j:
        #            print('"'+test[i]+' '+test[j]+'"'+' , '+'"'+test[j]+' '+test[i]+'",')

    

if __name__ == "__main__":
    main()
