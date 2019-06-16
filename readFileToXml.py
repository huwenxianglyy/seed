import  utils
import  os
import docObject
import uuid
from sklearn.model_selection import train_test_split
import numpy as np



# 读取 文本 保存到数据库
if __name__ == "__main__":

    file_root_path="E:/down/关系抽取/BioNLP-OST-2019_SeeDev-binary_train/BioNLP-ST-2016_SeeDev-binary_train/"

    result=[]
    for rt, dirs, files in os.walk(file_root_path):
        if len(files)>0:
            for f in files:
                if f.endswith("txt")!=True:
                    continue
                a1_file=os.path.splitext(f)[0]+".a1"
                a2_file=os.path.splitext(f)[0]+".a2"
                d = docObject.doc()
                tempLines = []
                d.doc_id = os.path.splitext(f)[0]
                lines=utils.readFile(os.path.join(rt,f))
                lineObjs = list(map(lambda x: docObject.line(x[0],d.doc_id,x[1]), enumerate(lines)))

                d.lines=lineObjs
                d.linesToWords()
                result.append(d)

    utils.loadData()
    # train, test = train_test_split(result, test_size=0.2)
    # utils.dumpData4Gb(train,"D:\\卡证要素提取\\data\\train1.bin")
    # utils.dumpData4Gb(test,"D:\\卡证要素提取\\data\\test1.bin")
    #
    # mysql_tool.save_docs(train,"1")
    # mysql_tool.save_docs(test,"0")


