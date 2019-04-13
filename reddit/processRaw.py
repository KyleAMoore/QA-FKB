"""

"""


import pickle
import re
from random import shuffle
from tqdm import tqdm
from sys import argv


def save(fname, *args):
    fout = open(fname, "wb")
    for obj in args:
        pickle.dump(obj, fout)
    fout.close()

def load(fname):
    fin = open(fname, "rb")
    obj = pickle.load(fin)
    fin.close()
    return obj

def process(posts):
    """
        resulting structure:

        ans = [
                {
                    "text" : ..., 
                    "id" : ...,
                    "relation" : ...
                }
              ]

        que = [
                {
                    "text" : ...,
                    "id" : ...,
                    "relation" : [...]
                }
              ]
    """

    que = []
    ans = []
    print("Cleaning and separating data")
    queID = 0
    for i, postDict in tqdm(posts.items()):
        queDict = {"id"       : queID,
                   "text"     : cleanString(postDict["text"]),
                   "relation" : [queID]}
        ansDict = {"id"       : queID,
                   "text"     : cleanString(postDict["answer"]),
                   "relation" : queID}
        queID += 1
        que.append(queDict)
        ans.append(ansDict)

    return que,ans

def cleanString(origStr):
    removePatt = re.compile("<.+?>|[^\s\w.?!]")       #for removing html tags and non-alphanum, non-terminating chars
    newStr = re.sub(removePatt, "", origStr)

    newStr = re.sub(r"[!\.]+", ".", newStr)          #removes duplicate . and !, converts ! to .
    newStr = re.sub(r"[\?\.]{2.}", " ?", newStr)     #removes duplicate ? and converts combination punctuation to ? (ex. ?!)
    newStr = re.sub(r"\.", " .", newStr)             #TODO: may be able to be done more efficiently/easily. #adds space before every period (eases tokenization)
    newStr = re.sub(r"\s+", " ", newStr)             #convert all whitespace to single space

    return newStr.lower()

def main():
    
    rawFile = argv[1] if len(argv) > 1 else "posts.pkl"
    try:
        posts = load(rawFile)
        que, ans = process(posts)
        save("data.pkl", que, ans)
    except FileNotFoundError:
        print(f"{rawFile} not found. Processing failed.")

if __name__ == '__main__':
    main()