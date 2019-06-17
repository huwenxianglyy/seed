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
# root_path="/home/huwenxiang/deeplearn/brat-v1.3_Crunchy_Frog/data/court/test"
# move_path="/home/huwenxiang/deeplearn/brat-v1.3_Crunchy_Frog/data/court/over/test" #笔录test
root_path="/home/huwenxiang/deeplearn/brat-v1.3_Crunchy_Frog/data/court/train2"
move_path="/home/huwenxiang/deeplearn/brat-v1.3_Crunchy_Frog/data/court/over/train2" #笔录 train
# root_path="/home/huwenxiang/deeplearn/brat-v1.3_Crunchy_Frog/data/court/qisuzhuang/test"
# move_path="/home/huwenxiang/deeplearn/brat-v1.3_Crunchy_Frog/data/court/qisuzhuang/over/test"

current_date=time.strftime('%Y-%m-%d',time.localtime(time.time()))
move_path=os.path.join(move_path,current_date)
if os.path.exists(move_path)==False:
    os.mkdir(move_path)
# 读取文件信息
file_size=utils.loadData("/home/huwenxiang/deeplearn/brat-v1.3_Crunchy_Frog/data/court/qisuzhuang/yubiao/file.size")
#
file_map={}# 这里 读取file size 文件。
for f in file_size:
    info=f.split("\t")
    file_map[info[0]]=info[1]

for rt, dirs, files in os.walk(root_path):
    files_filter=list(filter(lambda x:x.endswith(".ann"),files))
    for f in files_filter:# tqdm
        file_name=os.path.splitext(f)[0]
        file_ann=f
        file_txt=file_name+".txt"
        current_file_size=str(os.path.getsize(os.path.join(rt,file_ann)))
        # if file_name not in file_map.keys() :##  如果有预标，就用这个来判断
        #     continue
        # if file_map[file_name]==current_file_size: ## 如果有预标，就用这个来判断
        #     continue ##
        if current_file_size=="0":
            continue
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

                # try:  #这里的代码是用来找错的
                #     _,type_pos,text=e.split("\t")
                #     type, start, end = getType_pos(type_pos)  # 获取信息
                # except:
                #     print(""+ f)
                #     # shutil.copy(os.path.join(rt, f), os.path.join(move_path, f))
                #     break

                _,type_pos,text=e.split("\t")
                type,start,end=getType_pos(type_pos)# 获取信息
                sw=words[int(start)]
                ew=words[int(end)-1]
                # if sw.pos!=int(start) or ew.pos!=int(end)-1 or sw.text!=text[0] or ew.text!=text[-1]:
                #     print(""+f)
                #     # shutil.copy(os.path.join(rt, f), os.path.join(move_path, f))
                #     break  #这里的代码也是用来找错的
                assert sw.pos==int(start)
                assert ew.pos==int(end)-1
                assert sw.text==text[0]
                assert ew.text==text[-1]
                start_line_id=sw.line_id
                end_line_id=ew.line_id
                entityobj=docObject.entity(ei, text, start_line_id, end_line_id, start, end, type, file_name)
                entityobjs.append(entityobj)

            #这里 开始存入sql
            mysql_tool.save_entity(entityobjs) # 如果是那边标注得，就修改下is_check 1 是我们标注得，2 是他们标注得
            # 存入完毕， 将文件移走
            shutil.move(os.path.join(rt,f), os.path.join(move_path,f))# 转移ann 文件
            shutil.move(os.path.join(rt,file_txt), os.path.join(move_path,file_txt)) # 转移text 文件



