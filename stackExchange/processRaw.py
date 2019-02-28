"""
  #TODO: Adapt this code to work with new format and source
  #      does not need doc2vec (yet) and should not yet involve
  #      any excessive cleaning (lemmatization, etc.)
"""


import pickle
import re
from random import shuffle
from tqdm import tqdm


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

    newStr = re.sub(r"[!\.]+", " .", newStr)          #removes duplicate . and !  (space needed for tokenizer)
    newStr = re.sub(r"\?+|[\?\.]{2.}", " ?", newStr)  #removes duplicate ?
    newStr = re.sub(r"\s+", " ", newStr)              #convert all whitespace to single space
    #newStr = re.sub(r"[\?\.]{2,}", " ?", newStr)     #converts combination punctuation to ? (for catching things like ?!)

    return newStr.lower()

def tokenize(line):
    return line.split(" ")

def condense(questions, answers):
    answerRel = {} #maintains a temporary relation between previous ids and new id(i.e. index)
    qList = []
    print("Condensing Questions")
    for i,d in tqdm(questions.items()):
        d["id"] = len(qList)
        answerRel[i] = d["id"]
        d["relation"] = []
        qList.append(d)

    aList = []
    print("Condensing Answers")
    for i,d in tqdm(answers.items()):
        d["id"] = len(aList)
        parentId = answerRel[d["relation"]]
        d["relation"] = parentId
        if qList[parentId]["child"] != i: qList[parentId]["relation"].append(d["id"])
        aList.append(d)

    print("Ordering Answer Lists")
    # child being -1 indicates that there is no accepted answer => highest scored answer will be considered the accepted answer
    for que in tqdm(qList):
        que["relation"] = ([] if que["child"] == -1 else [que["child"]]) + sorted(que["relation"], key=(lambda x: aList[x]["score"]), reverse=True)
        del que["child"]
    
    return qList, aList

def main():
    fin = open("raw.csv", "r", errors="ignore", newline="\r\n") #because stackexchange ends csv lines with crlf and includes lf in body field
    lines = fin.readlines()
    fin.close()
    que, ans = condense(*toDict(lines[1:]))

    fout = open("data.pkl", "wb")
    pickle.dump(que, fout)
    pickle.dump(ans, fout)
    fout.close()

if __name__ == '__main__':
    main()