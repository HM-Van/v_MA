import numpy as np
# Information for all init configurations: obj on which table
envInfo=x = np.array([[1, 1, 2], #1 r g b
                        ],
                        np.int16)

test=["(held red)", "(on red green)", "(on red blue)", "(on red cyan)",
        "(held green)", "(on green red)", "(on green blue)", "(on green cyan)",
        "(held blue)", "(on blue red)", "(on blue green)", "(on blue cyan)",
        "(held cyan)", "(on cyan red)", "(on cyan green)", "(on cyan blue)"]

numSets=72
numEnv=99
# objectives consisting of 2 goal formulations
Sets=[
    "(held red) (held green)" , "(held red) (held green)",
    "(held red) (on green red)" , "(held red) (on green red)",
    "(held red) (on green blue)" , "(held red) (on green blue)",
    "(held red) (on green cyan)" , "(held red) (on green cyan)",
    "(held red) (held blue)" , "(held red) (held blue)",
    "(held red) (on blue red)" , "(held red) (on blue red)",
    "(held red) (on blue green)" , "(held red) (on blue green)",
    "(held red) (on blue cyan)" , "(held red) (on blue cyan)",
    "(held red) (held cyan)" , "(held red) (held cyan)",
    "(held red) (on cyan red)" , "(held red) (on cyan red)",
    "(held red) (on cyan green)" , "(held red) (on cyan green)",
    "(held red) (on cyan blue)" , "(held red) (on cyan blue)",
    "(on red green) (held green)" , "(on red green) (held green)",
    "(on red green) (on green blue)" , "(on red green) (on green blue)",
    "(on red green) (on green cyan)" , "(on red green) (on green cyan)",
    "(on red green) (held blue)" , "(on red green) (held blue)",
    "(on red green) (on blue red)" , "(on red green) (on blue red)",
    "(on red green) (on blue cyan)" , "(on red green) (on blue cyan)",
    "(on red green) (held cyan)" , "(on red green) (held cyan)",
    "(on red green) (on cyan red)" , "(on red green) (on cyan red)",
    "(on red green) (on cyan blue)" , "(on red green) (on cyan blue)",
    "(on red blue) (held green)" , "(on red blue) (held green)",
    "(on red blue) (on green red)" , "(on red blue) (on green red)",
    "(on red blue) (on green cyan)" , "(on red blue) (on green cyan)",
    "(on red blue) (held blue)" , "(on red blue) (held blue)",
    "(on red blue) (on blue green)" , "(on red blue) (on blue green)",
    "(on red blue) (on blue cyan)" , "(on red blue) (on blue cyan)",
    "(on red blue) (held cyan)" , "(on red blue) (held cyan)",
    "(on red blue) (on cyan red)" , "(on red blue) (on cyan red)",
    "(on red blue) (on cyan green)" , "(on red blue) (on cyan green)",
    "(on red cyan) (held green)" , "(on red cyan) (held green)",
    "(on red cyan) (on green red)" , "(on red cyan) (on green red)",
    "(on red cyan) (on green blue)" , "(on red cyan) (on green blue)",
    "(on red cyan) (held blue)" , "(on red cyan) (held blue)",
    "(on red cyan) (on blue red)" , "(on red cyan) (on blue red)",
    "(on red cyan) (on blue green)" , "(on red cyan) (on blue green)",
    "(on red cyan) (held cyan)" , "(on red cyan) (held cyan)",
    "(on red cyan) (on cyan green)" , "(on red cyan) (on cyan green)",
    "(on red cyan) (on cyan blue)" , "(on red cyan) (on cyan blue)",
    "(held green) (held blue)" , "(held green) (held blue)",
    "(held green) (on blue red)" , "(held green) (on blue red)",
    "(held green) (on blue green)" , "(held green) (on blue green)",
    "(held green) (on blue cyan)" , "(held green) (on blue cyan)",
    "(held green) (held cyan)" , "(held green) (held cyan)",
    "(held green) (on cyan red)" , "(held green) (on cyan red)",
    "(held green) (on cyan green)" , "(held green) (on cyan green)",
    "(held green) (on cyan blue)" , "(held green) (on cyan blue)",
    "(on green red) (held blue)" , "(on green red) (held blue)",
    "(on green red) (on blue green)" , "(on green red) (on blue green)",
    "(on green red) (on blue cyan)" , "(on green red) (on blue cyan)",
    "(on green red) (held cyan)" , "(on green red) (held cyan)",
    "(on green red) (on cyan green)" , "(on green red) (on cyan green)",
    "(on green red) (on cyan blue)" , "(on green red) (on cyan blue)",
    "(on green blue) (held blue)" , "(on green blue) (held blue)",
    "(on green blue) (on blue red)" , "(on green blue) (on blue red)",
    "(on green blue) (on blue cyan)" , "(on green blue) (on blue cyan)",
    "(on green blue) (held cyan)" , "(on green blue) (held cyan)",
    "(on green blue) (on cyan red)" , "(on green blue) (on cyan red)",
    "(on green blue) (on cyan green)" , "(on green blue) (on cyan green)",
    "(on green cyan) (held blue)" , "(on green cyan) (held blue)",
    "(on green cyan) (on blue red)" , "(on green cyan) (on blue red)",
    "(on green cyan) (on blue green)" , "(on green cyan) (on blue green)",
    "(on green cyan) (held cyan)" , "(on green cyan) (held cyan)",
    "(on green cyan) (on cyan red)" , "(on green cyan) (on cyan red)",
    "(on green cyan) (on cyan blue)" , "(on green cyan) (on cyan blue)",
    "(held blue) (held cyan)" , "(held blue) (held cyan)",
    "(held blue) (on cyan red)" , "(held blue) (on cyan red)",
    "(held blue) (on cyan green)" , "(held blue) (on cyan green)",
    "(held blue) (on cyan blue)" , "(held blue) (on cyan blue)",
    "(on blue red) (held cyan)" , "(on blue red) (held cyan)",
    "(on blue red) (on cyan green)" , "(on blue red) (on cyan green)",
    "(on blue red) (on cyan blue)" , "(on blue red) (on cyan blue)",
    "(on blue green) (held cyan)" , "(on blue green) (held cyan)",
    "(on blue green) (on cyan red)" , "(on blue green) (on cyan red)",
    "(on blue green) (on cyan blue)" , "(on blue green) (on cyan blue)",
    "(on blue cyan) (held cyan)" , "(on blue cyan) (held cyan)",
    "(on blue cyan) (on cyan red)" , "(on blue cyan) (on cyan red)",
    "(on blue cyan) (on cyan green)" , "(on blue cyan) (on cyan green)",
    "(on cyan green) (on cyan blue)" , "(on cyan green) (on cyan blue)"]


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

        for i in range(len(test)):
            for j in range(i,len(test)):
                if not i==j:
                    print('"'+test[i]+' '+test[j]+'"'+' , '+'"'+test[i]+' '+test[j]+'",')

    

if __name__ == "__main__":
    main()
