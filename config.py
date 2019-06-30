import utils
import numpy as np


def read_file(path):
    with open(path, "r", encoding='utf-8') as f1:
        data = f1.read().splitlines()
    return data

class ConfigParameter (object):
    def __init__(self):
        self.pre_embed = True
        self.num_epochs = 5000
        self.batch_size = 64
        self.sen_len = 156    # 最大句子长度
        self.max_position_len = 156 # 这个暂时没用。
        self.embedding_size = 300
        self.vocab_size = None
        self.hidden_size = 300
        self.pos_dim = 10
        self.pos_limit = 60    # 最大位置距离长度
        self.window = 3
        self.dropout = 0.5
        self.lr = 1e-4
        self.min_word_frequency = 2
        # self.embedding_file = 'E:/TOOLS/word2vec/eng_google/glove.6B.300d.txt'
        # self.embedding_file = 'd:/glove.6B.300d.txt'
        self.embedding_file = '/home/huwenxiang/deeplearn/词向量/英文/谷歌/glove.6B.300d.txt'
        self.train_data_path = '../bioNLP-SeeDev/BioNLP-ST-2016_SeeDev-binary_train'
        self.dev_data_path = '../bioNLP-SeeDev/BioNLP-ST-2016_SeeDev-binary_dev'
        self.save_data_path = './saved_data'
        self.entity_len=5# 这里是entity 最大长度
        self.relation_dim=50
        self.entity_type_dim=50

        # 这里加入词向量的操作
        self.PAD = 'PAD'
        self.UNK = 'UNK'

        # self.word2Vec_text = read_file(self.embedding_file)

        self.word_dim=300
        self.word2Vec, self.word2Id, self.Id2Word=utils.loadData("/home/huwenxiang/deeplearn/seed/saved_data/word2vec.bin")
        self.word2Vec=np.array(self.word2Vec,dtype=np.float32)


args = ConfigParameter()

config = {}

config["entity_type"] = ['Tissue', 'RNA', 'Protein_Family',
                         'Environmental_Factor', 'Genotype', 'Pathway',
                         'Protein', 'Regulatory_Network', 'Box', 'Protein_Complex',
                         'Gene_Family', 'Gene', 'Promoter', 'Hormone',
                         'Protein_Domain', 'Development_Phase']
config["entity_type2id"] = {t: i for i, t in enumerate(config["entity_type"])}
config["id2entity_type"] = {i: t for i, t in enumerate(config["entity_type"])}

config["entity1_relation_type"] = ['Participant', 'Agent', 'Domain', 'Molecule', 'Element1', 'Functional_Molecule',
                                   'Element', 'Source', 'Amino_Acid_Sequence', 'DNA_Part', 'Process', 'Agent1']
config["entity1_relation_type2id"] = {t: i for i, t in enumerate(config["entity1_relation_type"])}
config["id2entity1_relation_type"] = {i: t for i, t in enumerate(config["entity1_relation_type"])}

config["entity2_relation_type"] = ['Element2', 'Product', 'Development', 'DNA', 'Agent2', 'Target',
                                   'Genotype', 'Molecule', 'Family', 'Target_Tissue', 'Functional_Molecule',
                                   'Protein_Complex', 'Process']


config["entity_all_relation_type"]=["None",'Process', 'Domain', 'Family', 'DNA',
                                    'Target', 'DNA_Part', 'Molecule', 'Functional_Molecule',
                                    'Target_Tissue', 'Agent', 'Agent2', 'Protein_Complex',
                                    'Element1', 'Participant', 'Product', 'Development', 'Element',
                                    'Element2', 'Source', 'Agent1', 'Genotype', 'Amino_Acid_Sequence']


config["entity_all_relation_type2id"] = {t: i for i, t in enumerate(config["entity_all_relation_type"])}
config["id2entity_all_relation_type"] = {i: t for i, t in enumerate(config["entity_all_relation_type"])}






config["entity2_relation_type2id"] = {t: i for i, t in enumerate(config["entity2_relation_type"])}
config["id2entity2_relation_type"] = {i: t for i, t in enumerate(config["entity2_relation_type"])}

config["relation_type"] = ['None', 'Is_Involved_In_Process', 'Regulates_Expression',
                           'Has_Sequence_Identical_To','Exists_In_Genotype',
                           'Transcribes_Or_Translates_To', 'Occurs_In_Genotype', 'Composes_Protein_Complex',
                           'Is_Linked_To',
                           'Regulates_Process', 'Interacts_With', 'Binds_To', 'Occurs_During', 'Is_Member_Of_Family',
                           'Is_Functionally_Equivalent_To',
                           'Is_Localized_In', 'Regulates_Accumulation',
                           'Composes_Primary_Structure', 'Exists_At_Stage', 'Regulates_Development_Phase',
                           'Is_Protein_Domain_Of', 'Regulates_Tissue_Development', 'Regulates_Molecule_Activity']
config["relation_type2id"] = {t: i for i, t in enumerate(config["relation_type"])}
config["id2relation_type"] = {i: t for i, t in enumerate(config["relation_type"])}



