import os
import utils
import re
import collections


# typeMap=config["typeMap"]
class line(object):
    def __init__(self,line_id,doc_id,text):
        self.line_id=line_id
        self.doc_id=doc_id
        self.text=text


class doc(object):
    def __init__(self):
        self.doc_id=""
        self.type_id=""
        self.is_delete="0"
        self.is_check="0"
        self.file_name=""
        self.lines=[] # line 的list
        self.is_train="0"
        self.entitys={} # 这里是个Map 方便生成simple
        self.predict_entitys={} # 这里是个Map 方便生成simple
        self.words=[]
        self.simples=[]




    def linesToWords(self): # todo 这里通过 entity assert 下
        index=0
        for l in self.lines:
            l_texts=l.text
            line_id=l.line_id
            for w in l_texts:
               self.words.append(word(w,index,line_id))
               index+=1
            self.words.append(word("\n",index,line_id))
            index+=1
        if len(self.words)>0:# 这里居然有0kb的文件
            self.words.pop(-1)
        assert "\n".join(list(map(lambda x:x.text,self.lines)))=="".join(list(map(lambda x:x.text,self.words)))

    def  splitDocToSimple(self):# todo 这里在split的时候要保证 长度不大于 模型输入的长度，不然坐标可能会对不上
        tempWords=[]
        for w in self.words:
            if w.text in ["。","，","；","?","？","!","！"]:
                tempWords.append(w)
                if len(tempWords)>0:
                    self.creatSimple(list(tempWords))# 这里创建样本
                    tempWords.clear()
            else:
                tempWords.append(w)

        if len(tempWords)>0:
            self.creatSimple(list(tempWords))
            tempWords.clear()


    # def creatSimple(self,tempWords):#这里创建样本
    #     if len(tempWords)<FLAGS.max_seq_length-2:
    #         s = simple()
    #         s.words = tempWords
    #         self.simples.append(s)
    #     else:
    #         lines=self.getLinesFromWords(tempWords)
    #         self.createSimpleFromLines(lines)


    def createSimpleFromLines(self,lines):#在用标点无法划分的时候， 使用lines 来尝试划分
        tempWords=[]
        for l in lines:
            text="".join(list(map(lambda x:x.text,l)))
            search=re.search(":|：", text)
            if search is not None and search.start()<8:# 这里是分割点
                if len(tempWords)>0:
                    self.createSimpleFromLinesUtils(list(tempWords))
                    tempWords.clear()
                tempWords.extend(l)
            else:
                tempWords.extend(l)
        if len(tempWords)>0:
            self.createSimpleFromLinesUtils(list(tempWords))


    # def createSimpleFromLinesUtils(self,tempWords):# 这里最终生成 simple  如果 按照 : 分割还是大于 就采取截断的方式。
    #     if len(tempWords)<FLAGS.max_seq_length-2:
    #         s = simple()
    #         s.words = tempWords
    #         self.simples.append(s)
    #     else:# 如果按照冒号分割还是大于了长度，就用截断的方式
    #         split_count= int(len(tempWords)/(FLAGS.max_seq_length-2)) \
    #             if len(tempWords)%(FLAGS.max_seq_length-2) ==0 \
    #             else int(len(tempWords)/(FLAGS.max_seq_length-2) +1) # 先找出有几个分割点
    #         for i in range(split_count):# 遍历分割点， 去分割
    #             start=i*(FLAGS.max_seq_length-2)
    #             end=(i+1)*(FLAGS.max_seq_length-2)
    #             s = simple()
    #             s.words = tempWords[start:end]
    #             self.simples.append(s)


    def getLinesFromWords(self, tempWords):
        lines = []
        last_line_id = tempWords[0].line_id
        tempLines = []
        for w in tempWords:
            if w.line_id == last_line_id:
                tempLines.append(w)
            else:
                lines.append(list(tempLines))
                tempLines.clear()
                last_line_id = w.line_id
                tempLines.append(w)
        if len(tempLines) > 0:
            lines.append(list(tempLines))
            tempLines.clear()
        return lines
    def createSimpleId(self):
        for s in self.simples:
            s.id = self.doc_id + "-"+str(s.words[0].pos)+"-"+str(s.words[-1].pos)


    def getSimple(self):
        self.splitDocToSimple() # 分割获取simple 实例
        self.createSimpleId()#  创建 simple 的ID
        for s in self.simples:
            texts,entitys=self.convertSimpleWords(s.words)
            s.inputWords=self.removeWord(texts)
            s.labels=entitys

    def getOneSimple(self):# 这个是用mysql 中的数据生成 标注文件的时候使用。
        s = simple()
        s.words = self.words
        s.labels=list(self.entitys.values())
        self.simples.append(s)



    # # 这里假设我有分割 好的句子， 这里的分割就是单纯的分割  不会  去除任何  字符， 然后把句子变成训练样本。
    # def convertSimpleWords(self,words):# 这里的simpleLine 是一个list 里面是word元素 将来 转换 BIE 标签 也在这里转换 这里是用来训练用的
    #     texts = list(words) #
    #     entitys=[]
    #     for w in texts:
    #         w.label="O" # 初始化所有的w 注意这里的w 是simpale 的。
    #     for i,w in enumerate(texts):
    #         pos=w.pos
    #         if pos in self.entitys.keys() :# 这里使用的是全局的 entitys
    #             entity=self.entitys[pos]
    #             type=entity.type
    #             text=entity.text
    #             end=i+len(text)
    #             if type not in typeMap.keys():
    #                 continue # 这里如果遇到自定义的标注类型就直接跳过
    #             if end>len(texts):
    #                 continue # todo  这里后期找到好的分割方式后，就去掉， 这里如果遇到实体超出了句子范围，就继续。
    #             w.label="B-"+typeMap[type]
    #             for  last_i in range(i+1,end):
    #                 texts[last_i].label="I-"+typeMap[type]
    #             entitys.append(entity)
    #     return  texts,entitys # texts



    def removeWord(self,texts):# todo  复杂点的数字 转特殊字符
        texts=list(filter(lambda x:x.text!="\n" and x.text!=" " and x.text!="\t",texts))
        return texts


    def fromEntityToAnn(self,path):# 这里是给文件做 预标用的。
        index=0
        ann_strs=[]
        for s in self.simples:
            bucket={}
            for w in s.words:
                bucket[w.pos]=w.text # 创建一个桶，方便获取字符
            for e in s.predict_labels:
                start=e.start
                end=e.end
                ann_str="T"+str(index)+"\t"
                type_pos=[]# 位置信息
                text=""# 最终的text
                lasttext=""# 检车前一个字符是否是"\n"
                type_pos.append(str(start)) # 加入坐标的开始
                tempt_type_pos=""
                entity_type=e.type
                for i in range(start,end):
                    subt=bucket[i]# 拿到当前字符
                    if subt=="\n": # 如果当前字符是"\n"
                        if lasttext!="\n":# 如果之前不是"\n"，那么就是新的开始
                            tempt_type_pos+=str(i)+";"
                        text+=" "# "\n" 替换成 空格
                    else:
                        if lasttext=="\n": # 如果上一个字符 是"\n" 而当前字符不是 "\n" 那么就添加 新的开始位置
                            tempt_type_pos+=str(i)
                            type_pos.append(tempt_type_pos)
                            tempt_type_pos=""# 初始化为空
                        text+=subt
                    lasttext=subt
                type_pos.append(str(end))
                ann_str+=entity_type+" "+" ".join(type_pos)+"\t"+text
                ann_strs.append(ann_str)
                index+=1
        utils.saveToFile(os.path.join(path,self.doc_id+".ann"),ann_strs)





class simple(object):
    def __init__(self):
        self.id="" # Id 目前定义的是 doc_id +"-" + w.start+"-"+ w.end
        self.words=[] # docToSimple 方法用的words
        self.inputWords=[] # 这是丢入模型的words
        self.labels=[] # 这里里面应该是entity 对象 注意这里的entity 的坐标 都是全局的坐标
        self.predict_labels=[] # 这里里面应该是entity 对象 注意这里的entity 的坐标 都是全局的坐标

    # def setWordPredictLabel(self): # 这里吧预测的标签 写入 words
    #     for w in self.words:
    #         w.predict_label="O" # 初始化所有的w 注意这里的w 是simpale 的。
    #     predict_labels_map={ entity.start:entity for entity in self.predict_labels}
    #     for i,w in enumerate(self.words):
    #         pos=w.pos
    #         if pos in predict_labels_map.keys() :# 这里使用的是全局的 entitys
    #             entity=predict_labels_map[pos]
    #             type=entity.type
    #             # text=entity.text
    #             # end=i+len(text) # 这里文字可能对不上
    #             end=i+entity.end-entity.start
    #             if type not in typeMap.keys():
    #                 continue # 这里如果遇到自定义的标注类型就直接跳过
    #             w.predict_label="B-"+typeMap[type]
    #             for  last_i in range(i+1,end):
    #                 self.words[last_i].predict_label="I-"+typeMap[type]



# search_entity=config["search_entity"]# 是一个map
# search_entity_list=list(search_entity.keys())
# search_entity_pattern="|".join(list(search_entity.keys()))
# def searchEntityUseRule(samp):# 输入一个sample  search sample 下有没有 规则中出现的实体。
#     words=list(filter(lambda x:x.text!="\n" and x.text!=" " and x.text!="\t",samp.words)) # 先排除掉 \n 空格 制表符 这些字符
#     words_text="".join(list(map(lambda x:x.text,words)))
#     match_result=re.finditer(search_entity_pattern,words_text)
#     entity_list=[]
#     doc_id="-".join(samp.id.split("-")[0:-2])
#     for r in match_result:
#         start=r.start()# 这里是删除换行，空格后的 字符串位置
#         end=r.end()
#         text=r.group()
#         org_start=samp.words.index(words[start])
#         org_end=samp.words.index(words[end - 1])# 这里拿到的是当前词的坐标。如果不-1 就拿到的不对
#         org_text = "".join(list(map(lambda x: x.text, samp.words[org_start:org_end+1]))).replace("\n"," ")
#         ent = entity("1", org_text, words[start].line_id, words[end - 1].line_id,
#                                words[start].pos,  # 这里的entity id 设置为1， 在加入entity 的时候 重新设置。
#                                words[end - 1].pos + 1, search_entity[text], doc_id)
#         entity_list.append(ent)
#     return entity_list




def isExistEntity(bucket,start,end):# 返回真 说明这里已经有个实体， 否者 这里说明都没有
    for index in range(start, end):
        if bucket[index].predict_label != "O":
            return True
    return False

# def changeWordPredictLable(bucket,start,end,type):# 这里将规则判断的 label 改变到 word predict label中
#     bucket[start].predict_label="B-"+typeMap[type]
#     for index in range(start+1, end):
#         bucket[index].predict_label="I-"+typeMap[type]



# def entitysAppendRuleEntity(sample,rule_entitiys):# 向 entitys  中加入 rule_entity,并且对应的word predict_label 做出改变
#     bucket=collections.OrderedDict()
#     for w in sample.words:
#         bucket[w.pos] = w  # 创建一个桶，方便获取字符
#     for rule in rule_entitiys:
#         start=rule.start
#         end=rule.end# 注意这里的end 是结束词+1的位置。
#         type=rule.type
#         if isExistEntity(bucket,start,end)==False:
#             changeWordPredictLable(bucket,start,end,type)
#             sample.predict_labels.append(rule)# 这里rule 的id 需要确定下嘛，目前设计的是在预测的时候加入规则的信息。修复数据的时候可能需要重新设计。



class word(object):
    def __init__(self,text,pos,line_id):
        self.text=text
        self.pos=pos    # 这两个一定要是int 型
        self.line_id=line_id
        self.label=""
        self.predict_label=""




class entity(object):
    def __eq__(self, another):
        if type(self) == type(another) and self.type == another.type \
                and self.start == another.start and self.end == another.end :
            return True
        return False

    def __init__(self,id,text,start_line_id,end_line_id,start,end,type,doc_id):
        self.id=id
        self.text=text
        self.start_line_id=start_line_id # 这四个一定要是 int 型
        self.end_line_id=end_line_id
        self.start=start # 这里都是全局的 坐标
        self.end=end# 这里是全局的坐标 end 的坐标+1
        self.type=type
        self.doc_id=doc_id




