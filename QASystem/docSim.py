import pickle
from pathlib import Path
from time import time
from re import sub
#TODO: Lemmatization

#TODO: figure out when this would be necessary. May be beneficial
#TODO: to only import when needed due to long module load time
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def processDoc(doc):
    """
        #TODO:
        Processes the "raw" document to prepare it for the Doc2Vec model. This
        method should assume that basic cleaning (specifically that done by the
        cleanString function in each processRaw module) has been applied before
        calling this method. This method should convert this semi-clean input
        into the format expected by the Doc2Vec library. Specifically, output
        should have the following
        properties:
            * Lemmatized
            * Tokenized into word tokens
            * Tokenized into "sentence" tokens (may need to consider entire
              document as a "sentence")
            *#TODO: Add to this list as more requirements are identified!!!
    """
    pass

def createVectorizer():
    """
        #TODO:
        Creates (and trains?) the Doc2Vec Model
    """
    pass

def vectorize(doc):
    """
        #TODO:
        Should receive doc in a raw form (that is to say that the processDoc
        function should not be used before this function) and should result in
        a Doc2Vec representation for the document.
    """
    pass
    #likely only needs following line
    #return model.infer_vector(processDoc(doc))

def compare(doc1, doc2):
    """
        #TODO:
        Computes the "distance" between two documents. Likely should receive
        documents in a already vectorized representation, and will likely only
        be used within the findSim function.

        In its simplest form, this will likely use simple Euclidean distance or
        cosine similarity, but other distance metrics should be investigated if
        time allows.
    """
    pass

def findSim(doc, docList, k = 1):
    """
        #TODO:
        Finds the k most similar documents to doc that can be found in docList.
        For simplicity and efficiency, final version may use a document
        containing an indexed list of pre-vectorized representations of all
        training documents.

        Simplest method would be implementing k-nearest neighbor search, but
        that is computationally expensive. It may be worthwhile to attempt to
        find more efficient method. My only initial idea is to attempt some
        preproccesing, such as using a voronoi mapping. That, unfortunately,
        may be unusable for k > 1, and is inviable for a variable k.
    """
    pass

def preprocessTraining(data, outputFileName):
    """
        #TODO:
            This method should create and save pre-made versions of all
            documents in the training set. This is to both accelerate subsequent
            uses of this module. This will be especially important during the
            training of the Doc2Vec models.
    """
    pass

def main():
    pass

if __name__=="__main__":
    main()