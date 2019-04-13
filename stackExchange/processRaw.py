"""

"""


import pickle
import re
from random import shuffle
from tqdm import tqdm
from sys import argv


def save(obj, fname):
    fout = open(fname, "wb")
    pickle.dump(obj, fout)
    fout.close()

def load(fname):
    fin = open(fname, "rb")
    obj = pickle.load(fin)
    fin.close()
    return obj

def toDict(lines):
    """
        resulting structure:

        que = {
            id : {
                    "text" : ..., 
                    "score" : ...,
                    "relation" : ...
                 }
        }

        ans = {
            id : {
                    "text" : ...,
                    "child" : ...
                 }
        }
    """

    que = {}
    ans = {}
    print("Cleaning and separating data")
    for lin in tqdm(lines):
        #if lin != "\r\n":  #Ignores empty line (most likely at end of file)
            lineDict = {}
            postId = 0

            lin = lin[:-2] if lin.endswith("\r\n") else lin
            splitLin = lin.split(",")
            try:
                postId = int(splitLin[0][1:-1])
            except Exception as e:
                print(lin)
                raise e

            # recombination necessary becauase of commas in body
            # combines title and body together
            body = cleanString("".join(splitLin[2:-4]))
            
            try: #if no error, this post is an answer
                parentId = int(splitLin[-2][1:-1])
                score = int(splitLin[-1][1:-1])
                ans[postId] = {"text" : body, "score" : score, "relation" : parentId}
            except ValueError: #error here indicates that there is no parent (thus is a question)
                try:
                    child = int(splitLin[-3][1:-1])
                except ValueError: #no accepted answer given
                    child = -1
                que[postId] = {"text" : body, "child" : child}
    
    return que,ans

def cleanString(origStr):
    removePatt = re.compile("<.+?>|[^\s\w.?!]")       #for removing html tags and non-alphanum, non-terminating chars
    newStr = re.sub(removePatt, "", origStr)

    newStr = re.sub(r"[!\.]+", ".", newStr)          #removes duplicate . and !, converts ! to .
    newStr = re.sub(r"[\?\.]{2.}", " ?", newStr)     #removes duplicate ? and converts combination punctuation to ? (ex. ?!)
    newStr = re.sub(r"\.", " .", newStr)             #TODO: may be able to be done more efficiently/easily. #adds space before every period (eases tokenization)
    newStr = re.sub(r"\s+", " ", newStr)             #convert all whitespace to single space

    return newStr.lower()


def condense(questions, answers):
    """
        resulting structure:

        que = [
                {
                    "text" : ..., 
                    "score" : ...,
                    "relation" : ...
                }
              ]

        ans = [
                {
                    "text" : ...,
                    "relation" : [...]
                }
              ]
    """
    answerRel = {} #maintains a temporary relation between previous ids and new id(i.e. index)
    qList = []
    print("Condensing Questions")
    for i,d in tqdm(questions.items()):
        d["id"] = len(qList)
        answerRel[i] = d["id"]
        d["relation"] = []
        qList.append(d)

    aList = []
    questionRel = {}
    print("Condensing Answers")
    for i,d in tqdm(answers.items()):
        d["id"] = len(aList)
        questionRel[i] = d["id"]
        parentId = answerRel[d["relation"]]
        d["relation"] = parentId
        if qList[parentId]["child"] != i: qList[parentId]["relation"].append(d["id"])
        aList.append(d)

    print("Ordering Answer Lists")
    # child being -1 indicates that there is no accepted answer => highest scored answer will be considered the accepted answer
    for que in tqdm(qList):
        que["relation"] = ([] if que["child"] == -1 else [questionRel[que["child"]]]) + sorted(que["relation"], key=(lambda x: aList[x]["score"]), reverse=True)
        del que["child"]
    
    return qList, aList

def process(lines):
    return condense(*toDict(lines))

def main():
    
    rawFile = argv[1] if len(argv) > 1 else "raw.csv"
    try:
        fin = open(rawFile, "r", errors="ignore", newline="\r\n") #because stackexchange ends csv lines with crlf and includes lf in body field
        lines = fin.readlines()
        fin.close()
        que, ans = condense(*toDict(lines[1:]))

        fout = open("data.pkl", "wb")
        pickle.dump(que, fout)
        pickle.dump(ans, fout)
        fout.close()
    except FileNotFoundError:
        print(f"{rawFile} not found. Processing failed.")

if __name__ == '__main__':
    main()