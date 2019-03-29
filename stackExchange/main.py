import sys; sys.path.append("..\\")
from QASystem import docSim, summary
from random import shuffle
import processRaw
import rouge
import pickle

def saveData(filename, *args):
    fout = open(filename, "wb")
    for obj in args:
        pickle.dump(obj, fout)
    fout.close()

def loadData(filename):
    fin = open(filename, "rb")
    objs = []
    while True:
        try:
            objs.append(pickle.load(fin))
        except EOFError:
            break
    fin.close()
    return tuple(objs)

def trainTestSplit(questions, split=0.2):
    """
        return two distinct, but complete of integers representing the index of each question
    """
    indices = [i for i in range(len(questions))]
    shuffle(indices)

    return set(indices[int(split * len(questions)):]), set(indices[:int(split * len(questions))])    

def main(startStage = 1):
    """
        Workflow:
            ! 1. process raw data
            ! 2. split into train/test sets
                ! no need to remove answers from train set, only questions
            * for each question in train set:
                * 3. find 5 nearest questions in train set
                * extract top answer from each
                * 4. feed actual answer + 5 top answers into model (all at once)
            * 5. for each question in test set:
                * find 5 nearest questions in train set
                * extract top answer from each
                * feed top answers into model
                * compare result to actual answer
    """
    if(startStage == 1):
        fin = open("raw.csv", "r", errors="ignore", newline="\r\n") #because stackexchange ends csv lines with crlf and includes lf in body field
        lines = fin.readlines()
        fin.close()
        que, ans = processRaw.process(lines[1:])

        saveData("data.pkl", que, ans)
    
    #split sets here
    if(startStage <= 2):
        trainQues, testQues = trainTestSplit(que, split=0.2)
        print(trainQues, len(trainQues), end="\n\n")
        print(testQues, len(testQues))

    #find nearest questions here
    if(startStage <= 3):
        pass

    #train model
    if(startStage <= 4):
        pass

    #evaluate model
    if(startStage <= 5):
        pass

if __name__=="__main__":
    if len(sys.argv) == 1:
        startStage = 1
    else:
        startStage = int(sys.argv[1])
    main(startStage)