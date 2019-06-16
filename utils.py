



def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch = [], []
    for (x,y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        x_batch += [x]
        y_batch += [y]
    if len(x_batch) != 0:
        yield x_batch, y_batch






#对于ndarray 来处理
def minibatchesNdArray(test_ids, test_input_ids, test_input_masks, test_segment_ids, test_labels,words, minibatch_size):
    iterNum=len(test_ids)//minibatch_size
    for i in range(iterNum):
        start=i*minibatch_size
        end=(i+1)*minibatch_size
        yield test_ids[start:end],test_input_ids[start:end],test_input_masks[start:end],test_segment_ids[start:end],test_labels[start:end],words[start:end]





# -- coding: utf-8 --
import  json
import pickle
import re
import numpy as np
import jieba
import os
import shutil

def readMap(path):
    m={}
    with open(file=path, mode="r", encoding="utf-8") as f1:
        line=f1.read().splitlines();
        for l in line:
            key=l.split(" ")[0]
            value=l.split(" ")[1]
            m[key]=value
    return  m



'''
def readCSV(path):

'''


def readJson(path):
    jsonList=[]
    with open(file=path, mode="r", encoding="utf-8") as f1:
        line=f1.read().splitlines();
        for l in line:
            l=json.loads(l)
            jsonList.append(l)
    return  jsonList

# 判断当前的字符是否为中文 todo 要测试下是否对日文和韩文也是一样的
def _is_chinese_char(self, cp):
    """Checks whether CP is the codepoint of a CJK character."""
# This defines a "chinese character" as anything in the CJK Unicode block:
#   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
#
# Note that the CJK Unicode block is NOT all Japanese and Korean characters,
# despite its name. The modern Korean Hangul alphabet is a different block,
# as is Japanese Hiragana and Katakana. Those alphabets are used to write
# space-separated words, so they are not treated specially and handled
# like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

# 这个目前比较可靠
def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def splitDoc(doc):
    doc = " ".join(jieba.cut(doc))
    doc=re.sub(r"[0-9]{1,}\.{0,}[0-9]{0,}","",doc)
    #如果是word2vec训练的用下面这个
    # doc=re.sub(r"[0-9]{1,}\.{0,}[0-9]{0,}","&&&",doc)
    # doc=re.sub(r"[【】]","",doc)
    return  doc



import time
def jsonToMachineLearnData(jsonObjecTrain):
    docList = []
    docY = []
    for j in jsonObjecTrain:
        str=j["fact"]
        meta=j["meta"]
        relevant_articles=meta["relevant_articles"]
        if type(relevant_articles)==list:
            relevant_articles=relevant_articles[0]
        docList.append(str)
        docY.append(relevant_articles)
    return docList,docY

def dumpData( data,openPath):
    with open(openPath, "wb", ) as file:
        pickle.dump(data,file)

def dumpData4Gb(data,openPath):
    with open(openPath, "wb" ) as file:
        pickle.dump(data,file,protocol=4)

def loadData(openPath):
    with open(openPath, "rb", ) as file:
        data=pickle.load(file)
        return data

def removeNumber(obj):
    pattern=r"[0-9a-zA-Z]"
    l=[]
    for i in obj:
        search=re.search(pattern,i)
        if search !=None:
            continue
        l.append(i)
    return l




def csr_vappend(a,b):
    """ Takes in 2 csr_matrices and appends the second one to the bottom of the first one.
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""
    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])
    return a

def saveToFile(path,data,model="w"):
    with open(path,mode=model,encoding="utf-8") as f1:
        for l in data:
            f1.write(l+"\n")



def readFile(path,encod="utf-8"):
    with open(path,"r",encoding=encod) as f1:
        data=f1.read().splitlines()
    return data

def realLabelTransTrainLabel(label):
    label=list(label)
    realLabelTotrainLabel={}
    trainLabelToRealLabel={}
    for i, k in enumerate(label):
        realLabelTotrainLabel[k] = i
        trainLabelToRealLabel[i] = k
    dumpData("/home/huwenxiang/deeplearn/fayanbei/useData/realLabelTotrainLabel", realLabelTotrainLabel)
    dumpData("/home/huwenxiang/deeplearn/fayanbei/useData/trainLabelToRealLabel", trainLabelToRealLabel)


def maxSeqLengths(texts):
    max=0
    for line in texts:
        seq=len(line.split(" "))
        max=seq if seq>max else max
    return  max-1



def embedingInfo(text):
    wordToRowMap={}
    wordVec=[]
    for i,line in enumerate(text):
        line=line.strip().split()
        word=line[0]
        wordVec.append(line[1:])
        wordToRowMap[word]=i
    return np.array(wordVec,dtype=np.float32),wordToRowMap


def wordToIndexAndExpend(data,wordToRow,maxLength):
    result=np.zeros(shape=(len(data),maxLength))
    for row,line in enumerate(data):
        j=0
        for word in line:
            if word not in wordToRow:continue
            result[row,j]=wordToRow[word]
            j+=1
    return  result;


def removeDir(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except:
            os.remove(path)


def getStopWord(StopWodrpath):
    with open(StopWodrpath, 'r', encoding='utf-8') as f1:
        stopwords=set([line.strip() for line in f1.read().splitlines()] )
    return stopwords





def removeStopWords(sentence_seged):
    outstr = ''
    stopWordPath = "/home/huwenxiang/deeplearn/fayanbei/stopwords.txt"
    stopwords=getStopWord(stopWordPath)
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr


def mkdir(path,isDelete=False):
    # 引入模块
    import os
    import shutil


    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:

        os.makedirs(path)

        print(path + ' 创建成功')

        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        if(isDelete):
            shutil.rmtree(path)
            print(path + ' 删除成功')
            os.makedirs(path)
            print(path + ' 创建成功')

        else:
            print(path + ' 目录已存在')
        return False


def batch_iter(data, batch_size, shuffle=False):
    """
    Generates batches for the NN input feed.

    Returns a generator (yield) as the datasets are expected to be huge.
    """
    data = np.array(data)
    data_size = len(data)

    batches_per_epoch = data_size // batch_size

    # logging.info("Generating batches.. Total # of batches %d" % batches_per_epoch)

    if shuffle:
      indices = np.random.permutation(np.arange(data_size))
      shuffled_data = data[indices]
    else:
      shuffled_data = data
    for batch_num in range(batches_per_epoch):
      start_index = batch_num * batch_size
      end_index = min((batch_num + 1) * batch_size, data_size)
      yield shuffled_data[start_index:end_index]


# 把一行文字按间隔分割 , 一般来讲第一个是分类号
def split_line_by_space(line, delta=10):
    trim_line = [ch for ch in line if ch.x > -1]

    # 这里得到字间距
    spaces = [ch.x - (trim_line[i-1].x + trim_line[i-1].w) for i, ch in enumerate(trim_line) if i > 0]
    spaces.sort(reverse=1)# 字间距 按大到小排序
    deltas = [spaces[i-1] - sp for i, sp in enumerate(spaces) if i > 0] # 排序后的字间距，用前面的减去后面的
    max_delta = max(deltas)# 找出字间距变化最大的值
    if max_delta < delta:
        return [line]

    min_space = spaces[deltas.index(max_delta)]# 这里得到 哪一个字间距是分界点
    parts = []
    temp_chs = []
    for ch in trim_line:
        if len(temp_chs) == 0:
            temp_chs.append(ch)
            continue
        last_ch = temp_chs[-1]
        next_x = last_ch.x + last_ch.w + min_space # 这里预测下一个字的位置不应该会大于最大分割间距
        if ch.x < next_x:
            temp_chs.append(ch)
            continue
        else:
            if re.search("[;；，,:：]",last_ch.text) : # 这个条件主要是 '中图分类号：G482；X323' 这种会把；号当作分割点处理了
                temp_chs.append(ch)
                continue
            parts.append(temp_chs)
            temp_chs = [ch]

    if len(temp_chs) > 0:
        parts.append(temp_chs)

    return parts


def Jaccrad(model, reference):  # terms_reference为源句子，terms_model为候选句子
    grams_reference = set(reference)  # 去重；如果不需要就改为list
    grams_model = set(model)
    temp = 0
    for i in grams_reference:
        if i in grams_model:
            temp = temp + 1
    fenmu = len(grams_model) + len(grams_reference) - temp  # 并集
    jaccard_coefficient = float(temp / fenmu)  # 交集
    return jaccard_coefficient

