import utils
import numpy as np
wordorg=utils.loadData("./saved_data/train_word.bin")
word2=utils.loadData("./saved_data/dev_word.bin")
wordorg.update(word2)
PAD = 'PAD'
UNK = 'UNK'
word2Vec_text = utils.read_file('/home/huwenxiang/deeplearn/词向量/英文/谷歌/glove.6B.300d.txt')
word2Id = {}
Id2Word = {}
word2Vec = []
word_dim = 300

for i, w in enumerate(word2Vec_text):
    word, vec = w.split()[0], w.split()[1:]
    if word in wordorg:
        word2Id[word] = len(word2Vec)
        Id2Word[len(word2Vec)] = word
        word2Vec.append(vec)



word2Id[PAD] = len(word2Vec)
Id2Word[len(word2Vec)]=PAD
word2Vec.append([0] * 300)  # 这里用的300维的向量

word2Id[UNK] = len(word2Vec)
Id2Word[len(word2Vec)]=UNK
word2Vec.append(np.random.normal(0, 1, 300).tolist())


utils.dumpData4Gb((word2Vec,word2Id,Id2Word),"./saved_data/word2vec.bin")
