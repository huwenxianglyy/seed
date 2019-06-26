import utils
result=utils.loadData("./saved_data/train.bin")
result+=utils.loadData("./saved_data/dev.bin")

# 统计 类别
entity_type=set()
relation_type=set()
entity1_relation_type=set()
entity2_relation_type=set()
for d in result:
    for e in d.entities:
        entity_type.add(e.type)
    for r in d.relations:
        relation_type.add(r.relation_type)
        entity1_relation_type.add(r.entity1_type)
        entity2_relation_type.add(r.entity2_type)
print(1)

seq_len=[]
for d in result:
    for s in d.sentences:
        seq_len.append(len(s.text.split(" ")))

print(1)
