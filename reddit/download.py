import praw
import pandas
import datetime as dt
import pickle

def main():
    credFile = open("creds", "r")
    uname, pword, secret = credFile.read().split("\n")
    credFile.close()

    reddit = praw.Reddit(client_id="qc71466fJYq5ng",
                        client_secret=secret,
                        user_agent="QA-FKB-Download",
                        username=uname,
                        password=pword)

    askHist = reddit.subreddit("AskHistorians")

    top = askHist.top(time_filter="all", limit=1000)
    topYear = askHist.top(time_filter="year", limit=1000)
    hot = askHist.hot(limit=1000)
    contr = askHist.controversial(time_filter="all", limit=1000)

    posts = dict()
    posts = extract(top, posts)
    posts = extract(topYear, posts)
    posts = extract(hot, posts)
    posts = extract(contr, posts)
    
    print(len(posts))

    rawDict = dict()
    count = 0
    kept = 0
    for pid, post in posts.items():
        count += 1
        if post.num_comments > 0:
            try:
                postDict = dict()

                postDict["id"] = pid
                postDict["text"] = post.title + " " + post.selftext
                postDict["answer"] = post.comments[0].body

                rawDict[pid] = postDict
                kept += 1
            except IndexError: #probably caused by shadowbanned comments
                pass
        print(count, kept, end="\r")
        
    print()
    print(len(rawDict))

    fout = open("posts.pkl", "wb")
    pickle.dump(rawDict, fout)
    fout.close()

def extract(gen, postDict):
    for p in gen:
        postDict[p.id] = p
    return postDict

if __name__=="__main__":
    main()