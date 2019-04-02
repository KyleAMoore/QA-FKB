import pickle
from tqdm import tqdm
from pathlib import Path
from time import time
from re import sub
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, download
from nltk.corpus import wordnet, stopwords
from nltk.data import find
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def downloadNLTKData():
    try:
        find("corpora/wordnet")
    except LookupError:
        download("wordnet")
    try:
        find("corpora/stopwords")
    except:
        download("stopwords")
    try:
        find("taggers/averaged_perceptron_tagger")
    except:
        download('averaged_perceptron_tagger')

def saveModel(model, filename="vec.model"):
    model.save(filename)

def loadModel(filename="vec.model"):
    return Doc2Vec.load(filename)

def processDoc(doc, docID = -1):
    """
        Processes the "raw" document to prepare it for the Doc2Vec model. This
        method should assume that basic cleaning (specifically that done by the
        cleanString function in each processRaw module) has been applied before
        calling this method. This method should convert this semi-clean input
        into the format expected by the Doc2Vec library. Specifically, output
        should have the following
        properties:
            * Case-standardized
            * Stop words removed
            * Lemmatized
            * Tokenized into word tokens
            ! add to this list as more requirements are identified!!!

        docID defaults to -1, and should only be left as the default when the
        input document is being vectorized during the testing phase.
    """
    #tokenize and standardize case
    cleanDoc = (doc.lower()).split(" ")

    #remove stopwords (also removes empty strings from repeated spaces)
    stopWords = set(stopwords.words("english"))
    cleanDoc = [tok for tok in cleanDoc if tok not in stopWords and tok != ""]

    lem = WordNetLemmatizer()
    cleanDoc = pos_tag(cleanDoc)
    for tok in range(len(cleanDoc)):
        posTag = cleanDoc[tok][1][0]
        if   posTag == "J": posTag = wordnet.ADJ
        elif posTag == "V": posTag = wordnet.VERB
        elif posTag == "R": posTag = wordnet.ADV
        else:               posTag = wordnet.NOUN
        
        cleanDoc[tok] = lem.lemmatize(cleanDoc[tok][0], posTag)

    return cleanDoc

def createVectorizer(data):
    """
        Creates and trains the Doc2Vec Model

        Assumes data is in standardized form that results from processRaw
        modules.
    """
    print("Preparing Documents")
    clean = [TaggedDocument(processDoc(data[d]),[d]) for d in tqdm(range(len(data)))]

    print("Training Vectorizer")
    model = Doc2Vec(documents=clean,
                    vector_size=50,
                    window=5,
                    alpha=0.025,
                    min_alpha=0.0001,
                    min_count=5,
                    workers=10,
                    epochs=500) 
    
    #TODO: can be uncommented when model is definitely sufficiently designed and trained
    #model.delete_temporary_training_data()

    return model

def vectorize(model, doc):
    """
        !TEST
        Should receive doc in a raw form (that is to say that the processDoc
        function should not be used before this function) and should result in
        a Doc2Vec representation for the document..
    """
    return model.infer_vector(processDoc(doc),alpha=0.025,min_alpha=0.0001,epochs=5)

def compare(doc1, doc2):
    """
        #! May be best to use Doc2Vec.wv.n_similarity()
        Computes the "distance" between two documents. Likely should receive
        documents in a already vectorized representation, and will likely only
        be used within the findSim function.

        In its simplest form, this will likely use simple Euclidean distance or
        cosine similarity, but other distance metrics should be investigated if
        time allows. Regardless of method used, more similar docs should return
        a higher value than less similar docs
    """
    dist = 0
    for i in range(len(doc1)):
        dist += (doc1[i]-doc2[i]) ** 2
    return dist ** (1/2) if dist > 0 else 1000000000

def findSim(doc, docList, k = 1):
    """
        Finds the k most similar documents to doc that can be found in docList.
        For simplicity and efficiency, final version may use a document
        containing an indexed list of pre-vectorized representations of all
        training documents.

        Simplest method would be implementing k-nearest neighbor search, but
        that is computationally expensive. It may be worthwhile to attempt to
        find more efficient method. My only initial idea is to attempt some
        preproccesing, such as using a voronoi mapping. That, unfortunately,
        may be unusable for k > 1, and is inviable for a variable k.

        Returns the indices of the k most similar documents
    """
    docList = [(d,1/compare(doc,docList[d])) for d in range(len(docList))]
    docList.sort(key=lambda x: x[1],reverse=True)
    return [d[0] for d in docList[:k]]

def preprocessTraining(data, outputFileName):
    """
        !TEST
        This method should create and save pre-made versions of all documents
        in the training set. This is to both accelerate subsequent uses of this
        module. This will be especially important during the training of the
        Doc2Vec models.

        Should receive doc in a raw form (that is to say that the processDoc
        function should not be used before this function). Data should be in the
        standardized form that results from the processRaw modules.
    """
    vecs = [vectorize(data[d]["text"], d) for d in range(len(data))]
    fout = open(outputFileName, "wb")
    pickle.dump(vecs,fout)
    fout.close()

downloadNLTKData()