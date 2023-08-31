from multiprocessing import Value
import numpy as np
import cupy as cp
from gensim.models import Word2Vec
from gensim.test.utils import common_texts

# The Embedding model can be replaced by other embedding models
class EmbeddingBES36:
    BES36 = ['0,','1','2','3','4','5','6','7','8','9'
            ,'a','b','c','d','e','f','g','h','i','j'
            ,'k','l','m','n','o','p','q','r','s','t'
            ,'u','v','w','x','y','z']
    Dimensionality = 36

    BasicVectors = cp.diag(cp.ones(len(BES36)))
    def Embed(self,inputText):
        #Construct the value vector of a given value
        ValueVector = [0]*36 #36 for BES-36
        for character in inputText:
            try:
                i = self.BES36.index(character)
                tmpValue = ValueVector[i]
                tmpValue = tmpValue + 1
                ValueVector[i] = tmpValue
            except ValueError:
                continue
        tmpLength = np.linalg.norm(ValueVector)
        if(tmpLength == 0):
            NormalizedValueVector = ValueVector #Dealing with vector of all zeros
        else:
            NormalizedValueVector = ValueVector/tmpLength #Normalize a value vector to unit vector
        return NormalizedValueVector


class EmbeddingWord2Vec:
    Dimensionality = -1
    W2VModel = None
    BasicVectors = []
    def __init__(self,dim,AllTexts):
        self.Dimensionality = dim
        #word2vec.Word2Vec([s.encode('utf-8').split() for s in sentences]
        sentences=[s.split() for s in AllTexts]
        self.W2VModel = Word2Vec(sentences, vector_size=self.Dimensionality, window=10, min_count=1, workers=4)
        self.W2VModel.save("word2vec.model")
        self.W2VModel.train(AllTexts, total_examples=1, epochs=1)
        self.DefaultVector = np.ones(self.Dimensionality)
        self.BasicVectors = cp.diag(cp.ones(self.Dimensionality))

    def Embed(self,inputText):
        ValueVector = np.zeros(self.Dimensionality)
        for word in inputText.split():
            if self.W2VModel.wv.__contains__(word):
                ValueVector = ValueVector.__add__(self.W2VModel.wv[word])
            else:
                ValueVector = self.DefaultVector
        #Construct the value vector of a given value

        #for character in inputText:
        #    try:
        #        i = self.BES36.index(character)
        #        tmpValue = ValueVector[i]
        #        tmpValue = tmpValue + 1
        #        ValueVector[i] = tmpValue
        #    except ValueError:
        #        continue
        tmpLength = np.linalg.norm(ValueVector)
        if(tmpLength == 0):
            NormalizedValueVector = ValueVector #Dealing with vector of all zeros
        else:
            NormalizedValueVector = ValueVector/tmpLength #Normalize a value vector to unit vector
        return NormalizedValueVector