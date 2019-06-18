import utils
import os
import docObject
import mysql_tool
import shutil
from tqdm  import  tqdm

def getType_pos(type_pos):
    t_p=type_pos.split(" ")
    assert len(t_p)>=3
    type=t_p[0]
    start=t_p[1]
    end=t_p[-1]
    return type,start,end

# 将标注好的数据 保存到数据库，标注数据转移


import time

root_path="/home/huwenxiang/deeplearn/brat-v1.3_Crunchy_Frog/data/court/train2"

for rt, dirs, files in os.walk(root_path):
    files_filter=list(filter(lambda x:x.endswith(".ann"),files))
    for f in files_filter:#
        file_name=os.path.splitext(f)[0]
        file_ann=f
        file_txt=file_name+".txt"


        with open(os.path.join(rt,f)) as f1: #
            doc=f1.read()
        with open(os.path.join(rt,file_ann)) as f2 :
            entitys=f2.read().splitlines() # 读取entitys
        entitys=list(filter(lambda x:x.strip()!="",entitys))



        if len(entitys)>0:
            doc=mysql_tool.load_docs(doc_id=file_name)[0]
            words=doc.words
            entityobjs=[]
            for ei,e in enumerate(entitys):

                _,type_pos,text=e.split("\t")
                type,start,end=getType_pos(type_pos)# 获取信息
                sw=words[int(start)]
                ew=words[int(end)-1]
                assert sw.pos==int(start)
                assert ew.pos==int(end)-1
                assert sw.text==text[0]
                assert ew.text==text[-1]
                start_line_id=sw.line_id
                end_line_id=ew.line_id
                entityobj=docObject.entity(ei, text, start_line_id, end_line_id, start, end, type, file_name)
                entityobjs.append(entityobj)





