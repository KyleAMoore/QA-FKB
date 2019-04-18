import sys; sys.path.append("..\\")
from QASystem import docSim, summary
from random import shuffle, randint
from tqdm import tqdm
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
    return tuple(objs) if len(objs) > 1 else objs[0]

def trainTestSplit(questions, split=0.2):
    """
        return two distinct, but complete of integers representing the index of each question
    """
    indices = [i for i in range(len(questions))]
    shuffle(indices)

    return indices[int(split * len(indices)):], indices[:int(split * len(indices))] 

def main(startStage = 1):
    """
        Workflow:
            ! 1. process raw data
            ! 2. split into train/test sets
                ! no need to remove answers from train set, only questions
            ! 3. create vectorizer
            ! for each question in train set:
                ! 4. find 5 nearest questions in train set
                ! extract top answer from each
                ! 5. feed actual answer + 5 top answers into model (all at once)
            * for each question in test set:
                ! 6. find 5 nearest questions in train set
                ! extract top answer from each
                ! 7. feed top answers into model
                ! 8. compare result to actual answer
                ! save results to disk
    """
    print("Processing Raw Data")
    if(startStage == 1):
        fin = open("raw.csv", "r", errors="ignore", newline="\r\n") #because stackexchange ends csv lines with crlf and includes lf in body field
        lines = fin.readlines()
        fin.close()
        que, ans = processRaw.process(lines[1:])

        saveData("data.pkl", que, ans)
    else:
        que, ans = loadData("data.pkl")

    #split sets here
    print("\nSplitting Training and Testing Set")
    if(startStage <= 2):
        trainQues, testQues = trainTestSplit(que, split=0.2)
        saveData("split.pkl", trainQues, testQues)
    else:
        trainQues, testQues = loadData("split.pkl")

    #find nearest questions here
    print("\nTraining Similar Question Selection Model")
    if(startStage <= 3):
        docVec = docSim.createVectorizer([que[i]["text"] for i in trainQues])
        docSim.saveModel(docVec, "seVec.model")
    else:
        docVec = docSim.loadModel("seVec.model")
    
    #extract answers
    numSim = 5
    print("\nExtracting Candidate Answers in Training Set")
    if(startStage <= 4):
        queText = [que[i]["text"] for i in trainQues]
        simQueList = []

        queVectors = [docSim.vectorize(docVec, q) for q in tqdm(queText)]

        """
            should result in a data structure that lines up with trainQues such that the
            question at que[trainQues[i]] is most similar to simQueList[i]
        """
        for q in tqdm(range(len(queText))):
            simQueList.append(docSim.findSim(queVectors[q], queVectors, numSim))
        
        """
            result: each element is a list of 6 strings. The first string is the "correct" answer
                    and each subsequent string are the answers to similar questions 
        """
        ansList = [
                    [ans[que[origQ]["relation"][0]]["text"]] +          
                    [
                        ans[que[simQ]["relation"][0]]["text"]
                    for simQ in simQueList[origQ]]
                  for origQ in range(len(simQueList))]

        saveData("trainInput.pkl", ansList, queVectors)
    else:
        ansList, queVectors = loadData("trainInput.pkl")
    
    #train model
    vocabSize = 5000
    ansLen = 100
    wordEmbDim = 128
    contextVecLen = 128
    batchSize = 32
    epochs = 200
    validationSplit = 0.2
    print("\nTraining Summarizer Model")
    if(startStage <= 5):
        summarizer = summary.createModel(vocabSize, numSim*ansLen, ansLen, wordEmbDim, contextVecLen)
        summarizer, tokenizer = summary.trainModel(summarizer, ansList, vocabSize, ansLen, batchSize, epochs, validationSplit, fileName="seSum", cv=True, es=True, halfInpCV=True)
        summary.saveModel(summarizer, tokenizer, "seSum")
    else:
        summarizer, tokenizer = summary.loadModel("seSum")


    #find nearest questions to test set
    print("\nFinding Similar Questions in Testing Set")
    if(startStage <= 6):
        testQText = [que[i]["text"] for i in testQues]
        actAnswers = [ans[que[i]["relation"][0]]["text"] for i in testQues]
        simQueList = []
        
        testQVectors = [docSim.vectorize(docVec, q) for q in tqdm(testQText)]

        for q in tqdm(range(len(testQText))):
            simQueList.append(docSim.findSim(testQVectors[q], queVectors, numSim))

        testAnsList = [
                        [
                            ans[que[simQ]["relation"][0]]["text"]
                        for simQ in origQ]
                      for origQ in simQueList]

        saveData("testInput.pkl", testAnsList, testQText, actAnswers)
    else:
        testAnsList, testQText, actAnswers = loadData("testInput.pkl")

    #generate test summaries
    print("\nGenerating Answers for Testing Questions")
    if(startStage <= 7):
        summaries = summary.generateSummaries(summarizer, testAnsList, tokenizer, ansLen, batchSize)
        saveData("summaries.pkl", summaries)
    else:
        summaries = loadData("summaries.pkl")


    #evaluate model
    print("\nEvaluating Generated Answers")
    if(startStage <= 8):
        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                                max_n=3,
                                limit_length=True,
                                length_limit=100,
                                length_limit_type='words',
                                apply_avg=True,
                                apply_best=False,
                                alpha=0.5,
                                weight_factor=1.2,
                                stemming=True)

        scoresAvg = evaluator.get_scores(summaries, actAnswers)
        saveData("scores.pkl", scoresAvg)
    else:
        scoresAvg = loadData("scores.pkl")
    
    outf = open("results/rouge_avg", "w")
    outf.write(formatScores(scoresAvg, "average"))
    outf.close()

    testQText = [que[i]["text"] for i in testQues]

    for i, res in sampleResults(testQText, actAnswers, summaries).items():
        outf = open(f"results/qa-trip_{i}", "w")
        outf.write(f"{testQText[i]}\n\n{actAnswers[i]}\n\n{summaries[i]}")
        outf.close()

def sampleResults(questions, actAns, genAns, n=20):
    ind = set()
    while len(ind) < n:
        ind.add(randint(0,len(questions)-1))

    samples = {}
    for i in ind:
        samples[i] = {'q' : questions[i], 'a' : actAns[i], 'r' : genAns[i]}
    
    return samples

def formatScores(scores, label):
    retStr = ""
    retStr += ' ' + formatItem(label, 91, '-') + "\n"
    retStr += "|        metric        |       f1-score       |      precision       |        recall        |" + "\n"
    retStr += '|' + ('-' * 91) + '|' + "\n"
    for met, metScores in scores.items():
        retStr += '|' + "|".join([formatItem(met), formatItem(metScores["f"]), formatItem(metScores["p"]), formatItem(metScores["r"])]) + "|\n"
    retStr += ' ' + ('-' * 91)

    return retStr

def formatItem(cont, length=22, filler=' '):
    cont = str(cont)
    cont = (filler * (int(length / 2) - int(len(cont) / 2))) + cont
    cont += filler * (length - len(cont))
    return cont

if __name__=="__main__":
    if len(sys.argv) == 1:
        startStage = 1
    else:
        startStage = int(sys.argv[1])
    main(int(sys.argv[1]) if len(sys.argv) == 2 else 1)