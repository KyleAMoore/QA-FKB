from keras_self_attention import SeqSelfAttention
from keras.models import Model, load_model, model_from_json
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, TimeDistributed, Softmax, Lambda, Flatten
from keras.regularizers import l1, l2
from keras.preprocessing import text, sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping
from numpy import array as nparray
from numpy import zeros, concatenate
from keras_preprocessing.text import tokenizer_from_json
import pickle
import random
import gc
from tqdm import tqdm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

def permute(tensor):
    from keras.backend import permute_dimensions
    return permute_dimensions(tensor, (0,2,1))

def createModel(vocabSize, srcLength=500, sumLength=100, wordEmbDim=128, contextVecLen=128):
    #sourcetxt input
    inputs = (Input(shape=(srcLength,)))
    emb = Embedding(vocabSize, wordEmbDim, mask_zero=True)(inputs)
    encLSTM = Bidirectional(LSTM(units=contextVecLen, return_sequences=True))(emb)

    att = SeqSelfAttention(attention_width=10,
                           attention_activation="sigmoid",
                           attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                           kernel_regularizer=l2(1e-4),
                           bias_regularizer=l1(1e-4),
                           attention_regularizer_weight=1e-4,
                           name="Attn")(encLSTM)

    trnsp1 = Lambda(permute)(att)
    condense = Dense(100)(trnsp1)
    trnsp2 = Lambda(permute)(condense)

    #decoder output
    decLSTM = LSTM(units=contextVecLen, return_sequences=True)(trnsp2)
    dense = TimeDistributed(Dense(vocabSize, activation='relu'))(decLSTM)
    sftmx = TimeDistributed(Dense(vocabSize, activation='softmax'))(dense)
    
    #encoder+decoder
    model = Model(inputs=inputs, outputs=sftmx)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

def trainModel(model, inputs, vocabSize, ansLen=100, batch_size = 32, epochs = 200, validation_split=0.2, fileName="summarizer", cv=False, es=False):
    """
        Trains the model
        * inputs should be a list of answer groups. Each answer group should be
          represented by a list of strings where the first string is the accepted answer

        returns: model and tokenizer
    """
    print("Creating Tokenizer")
    tok = text.Tokenizer(vocabSize-1, lower=False, oov_token="UNK")
    tok.fit_on_texts(ans for ansGroup in inputs for ans in ansGroup)
    
    fout = open(fileName+'.tok', 'wb')
    pickle.dump(tok.to_json(), fout)
    fout.close()
    
    print("Preparing training data")
    inputAns = []
    outputAns = []
    for ansGroup in tqdm(inputs):
        numAnswers = len(ansGroup) - 1
        tokAns = tok.texts_to_sequences(ansGroup)
        #restrict to 100 tokens from each anwer and concatenate input answers together
        inp = [w for seq in tokAns[1:] for w in seq[:ansLen]]
        outp = tokAns[0][:ansLen]

        inputAns.append(inp)
        outputAns.append(outp)

    print("Padding/trimming inputs")
    inputAns = sequence.pad_sequences(inputAns, maxlen=ansLen*numAnswers, padding="post", truncating="post")
    outputAns = sequence.pad_sequences(outputAns, maxlen=ansLen, padding="post", truncating="post")
    
    def f(i):
        x = [0] * vocabSize
        x[i] = 1
        return x
    print("Finalizing training output")
    outNP = zeros((len(outputAns), ansLen, vocabSize))
    for i, doc in enumerate(tqdm(outputAns)):
        for j, word in enumerate(doc):
            outNP[i][j][word] = 1
    outputAns = outNP
    
    if(cv):
        print("Performing Cross-Validation")
        scores = crossValidate(model.to_json(), inputAns[:10], outputAns[:10])
        print(scores)
        scoreFile = open(fileName+".cvscores", "wb")
        pickle.dump(scores, scoreFile)
        scoreFile.close()
    
    print("Training Model")
    callbacks=[ModelCheckpoint(fileName+"{epoch:02d}_{loss:.2f}_{val_loss:.2f}.model", verbose=1, period=10)]
    if(es):
        callbacks.append(EarlyStopping(monitor="val_loss",
                                       patience=2,
                                       verbose=1,
                                       mode="min",
                                       restore_best_weights=True))
    
    model.fit(inputAns,
              outputAns,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=validation_split, 
              callbacks=callbacks)

    return model, tok

def generateSummaries(model, inputs, tokenizer, ansLen=100, batch_size=32,):
    """
        Generates summaries
        * inputs should be a list of answer groups. Each answer group should be
          represented by a list of strings
    """
    inputAns = []
    for ansGroup in inputs:
        numAnswers = len(ansGroup)
        tokAns = tokenizer.texts_to_sequences(ansGroup)
        #restrict to 100 tokens from each anwer and concatenate input answers together
        inp = [w for seq in tokAns for w in seq[:ansLen]]
        
        inputAns.append(inp)
    inputAns = sequence.pad_sequences(inputAns, maxlen=ansLen*numAnswers, padding="post", truncating="post")
    print(*inputAns, sep='\n')
    tokSumm = model.predict(nparray(inputAns), batch_size=batch_size, verbose=1)
    #removes padding tokens before conversion
    print(tokSumm)
    tokSum = [sample(summ) for summ in tokSumm]
    summaries = tokenizer.sequences_to_texts(tokSumm)

    return summaries

def crossValidate(modelJSON, input, output, epochs=5, batch_size=32, verbose=1, k=5):
    indices = [(int(len(input)*(i/k)),int(len(input) *((i+1)/k)))  for i in range(k)]
    losses = []
    count = 1
    for start,stop in indices:
        print(f"Cross validation stage {str(count)} out of {k}")
        count += 1
        tempInp = concatenate([input[0:start], input[stop:len(input)], input[start:stop]])
        tempOut = concatenate([output[0:start], output[stop:len(input)], output[start:stop]])
        
        model = model_from_json(modelJSON, custom_objects={'SeqSelfAttention':SeqSelfAttention})
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        hist = model.fit(tempInp, tempOut, epochs=epochs, verbose=verbose, validation_split=int(1/k))
        losses.append(hist)
        
        del(model)
        del(tempInp)
        del(tempOut)
        gc.collect()
        
    fout = open("hist", "wb")
    pickle.dump(losses, fout)
    fout.close()
    return hist
    
def sample(vector):
    ind = random.random(0,1)
    prob = 0
    for p in range(len(vector)):
        prob += vector[p]
        if prob >= ind: return p

def saveModel(model, tokenizer, name='qasystem'):
    model.save(name+'.model')
    fout = open(name+'.tok', 'wb')
    pickle.dump(tokenizer.to_json(), fout)
    fout.close()

def loadModel(name='qasystem'):
    fin = open(name+'.tok', 'rb')
    tokJSON = pickle.load(fin)
    fin.close()

    return load_model(name+'.model', custom_objects={'SeqSelfAttention':SeqSelfAttention}), tokenizer_from_json(tokJSON)

if __name__ == "__main__":
    k = 5
    summaryLength = 100
    wordEmbDim = 128
    vocabSize = 3000
    contextVecLen = 128

    model = createModel(vocabSize, k*summaryLength, summaryLength, wordEmbDim, contextVecLen)