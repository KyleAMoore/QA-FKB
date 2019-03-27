from keras_self_attention import SeqSelfAttention
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional
from keras.backend import concatenate
from keras.regularizers import l1, l2
import pickle

def createModel():
    vocabSize = 1000
    srcLength = 500
    sumLength = 100
    wordEmbDim = 128

    model = Sequential()

    #sourcetxt input
    inputs = (Input(shape=(srcLength,)))
    emb = Embedding(vocabSize, wordEmbDim)(inputs)
    lstm = Bidirectional(LSTM(units=wordEmbDim, return_sequences=True))(emb)

    att = SeqSelfAttention(attention_width=10,
                           attention_activation="sigmoid",
                           attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                           kernel_regularizer=l2(1e-4),
                           bias_regularizer=l1(1e-4),
                           attention_regularizer_weight=1e-4,
                           name="Attn")(lstm)

    #decoder output
    dense = Dense(vocabSize, activation='softmax')(att)
    model = Model(inputs=inputs, outputs=[dense])

    #encoder+decoder
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

if __name__ == "__main__":
    model = createModel()