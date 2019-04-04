from keras_self_attention import SeqSelfAttention
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, TimeDistributed, Softmax, Lambda, Flatten
from keras.regularizers import l1, l2
from keras.preprocessing import text, sequence
from tensorflow.dtypes import int64, float32, cast
from tensorflow.math import argmax
from tensorflow.random import categorical
from numpy import array as nparray
import keras.backend
import pickle

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

    trnsp1 = Lambda(lambda x: keras.backend.permute_dimensions(x, (0,2,1)))(att)
    condense = Dense(100)(trnsp1)
    trnsp2 = Lambda(lambda x: keras.backend.permute_dimensions(x, (0,2,1)))(condense)

    #decoder output
    decLSTM = LSTM(units=contextVecLen, return_sequences=True)(trnsp2)
    dense = TimeDistributed(Dense(vocabSize, activation='relu'))(decLSTM)
    sample = TimeDistributed(Dense(1, activation='softmax'))(dense)
    flat = Flatten()(sample)

    #encoder+decoder
    model = Model(inputs=inputs, outputs=flat)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    model.ansLen = sumLength

    return model

def trainModel(model, inputs, vocabSize, batch_size = 32, epochs = 1000, validation_split=0.2):
    """
        Trains the model
        * inputs should be a list of answer groups. Each answer group should be
          represented by a list of strings where the first string is the accepted answer

        returns: model and tokenizer
    """
    tok = text.Tokenizer(vocabSize-1, lower=False, oov_token="UNK")
    tok.fit_on_texts(ans for ansGroup in inputs for ans in ansGroup)
    
    inputAns = []
    outputAns = []
    for ansGroup in inputs:
        numAnswers = len(ansGroup) - 1
        tokAns = tok.texts_to_sequences(ansGroup)
        #restrict to 100 tokens from each anwer and concatenate input answers together
        inp = [w for seq in tokAns[1:] for w in seq[:model.ansLen]]
        outp = tokAns[0][:model.ansLen]

        #pad sequences 
        # inp += [-1] * ((model.ansLen * numAnswers) - len(inp))
        # outp += [-1] * (model.ansLen - len(outp))

        inputAns.append(inp)
        outputAns.append(outp)

    inputAns = sequence.pad_sequences(inputAns, maxlen=model.ansLen*numAnswers, padding="post", truncating="post")
    outputAns = sequence.pad_sequences(outputAns, maxlen=model.ansLen, padding="post", truncating="post")
    model.fit(inputAns, outputAns, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=validation_split)

    return model, tok

def generateSummaries(model, inputs, tokenizer, batch_size=32):
    """
        Generates summaries
        * inputs should be a list of answer groups. Each answer group should be
          represented by a list of strings
    """
    inputAns = []
    for ansGroup in inputs:
        numAnswers = len(ansGroup)
        tokAns = tok.texts_to_sequences(ansGroup)
        #restrict to 100 tokens from each anwer and concatenate input answers together
        inp = [w for seq in tokAns for w in seq[:model.ansLen]]

        #pad sequences 
        inp += [-1] * ((model.ansLen * numAnswers) - len(inp))

        inputAns.append(inp)

    tokSumm = model.predict(nparray(inputAns), batch_size=batch_size, verbose=1)
    #removes padding tokens before conversion
    summaries = tokenizer.sequences_to_texts([i for i in seq if i <= -1] for seq in tokSumm)

    return summaries

def saveModel(model, tokenizer, name='qasystem'):
    model.save(name+'.model')
    fout = open(name+'.tok', 'wb')
    pickle.dump(tokenizer.to_json, fout)
    fout.close()

def loadModel(name='qasystem'):
    fin = open(name+'.tok', 'rb')
    tokJSON = pickle.load(fin)
    fin.close()

    return load_model(name+'.model'), Tokenizer.tokenizer_from_json(tokJSON)

if __name__ == "__main__":
    k = 5
    summaryLength = 100
    wordEmbDim = 128
    vocabSize = 3000
    contextVecLen = 128

    model = createModel(vocabSize, k*summaryLength, summaryLength, wordEmbDim, contextVecLen)