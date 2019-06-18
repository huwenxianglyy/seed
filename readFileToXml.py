import  utils
import  os
import docObject
import uuid
from sklearn.model_selection import train_test_split
import numpy as np

def getType_pos(type_pos):# 这里获取 实体类型和实体位置 todo  这里获取信息 要注意 标签得单词之间有空格得情况  可以判断 t_p得长度，根据不同长度做处理
    t_p=type_pos.split(" ")
    assert len(t_p)>=3
    type=t_p[0]
    start=t_p[1]
    end=t_p[-1]
    return type,start,end




# 读取 文本 保存到数据库
if __name__ == "__main__":

    file_root_path="E:/down/关系抽取/BioNLP-OST-2019_SeeDev-binary_train/BioNLP-ST-2016_SeeDev-binary_train/"

    result=[]
    for rt, dirs, files in os.walk(file_root_path):# os.walk 会循环遍历文档下的所有文件 rt是根目录，dirs 是当前目录下的所有文件夹的名字，files是当前目录下所有文件的名字
        if len(files)>0:
            for f in files:
                if f.endswith("txt")!=True:
                    continue
                a1_file=os.path.splitext(f)[0]+".a1"#
                a2_file=os.path.splitext(f)[0]+".a2"# 这里获取 两个标注文件

                d = docObject.doc()
                tempLines = []
                d.doc_id = os.path.splitext(f)[0]# doc的ID就用文件名来代替
                lines=utils.readFile(os.path.join(rt,f))# 读取文件内容，是一个list ，每个元素对应原文中的一行
                lineObjs = list(map(lambda x: docObject.line(x[0],d.doc_id,x[1]), enumerate(lines)))# 这里 使用map 将 lines 变成 docObject.line 对象
                d.lines=lineObjs
                d.linesToWords()
                # 上面 doc 对象创建完成

                with open(os.path.join(rt, a1_file)) as f2:  # 读取实体标注文件
                    entitys = f2.read().splitlines()  # 读取entitys
                entitys = list(filter(lambda x: x.strip() != "", entitys)) # 去掉空行

                if len(entitys) > 0:
                    words = d.words
                    entityobjs = []
                    for eid, e in enumerate(entitys):
                        _, type_pos, text = e.split("\t")# 在brat 中 标注格式是t1\t类型 坐标 坐标\t正文  所以先用\t分割
                        type, start, end = getType_pos(type_pos)  # 获取信息
                        sw = words[int(start)] # 这里以及下面的assert 主要是校验用的，看标注文件标注的实体和原文中实体是不是一至的。
                        ew = words[int(end) - 1]
                        assert sw.pos == int(start)
                        assert ew.pos == int(end) - 1
                        assert sw.text == text[0]
                        assert ew.text == text[-1]
                        start_line_id = sw.line_id
                        end_line_id = ew.line_id
                        entityobj = docObject.entity(eid, text, start_line_id, end_line_id, start, end, type, d.doc_id)
                        d.entitys.append(entityobj)# 保存到当前doc对象得entitys中

                result.append(d)

    utils.dumpData4Gb(result, "D:\\卡证要素提取\\data\\train1.bin") # 将文件保存在需要的路径中



