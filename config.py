config={}
config["entity_type"]=['Tissue', 'RNA', 'Protein_Family',
                       'Environmental_Factor', 'Genotype', 'Pathway',
                       'Protein', 'Regulatory_Network', 'Box', 'Protein_Complex',
                       'Gene_Family', 'Gene', 'Promoter', 'Hormone',
                       'Protein_Domain', 'Development_Phase']
config["entity_type2id"]={t:i for i,t in  enumerate(config["entity_type"])}
config["id2entity_type"]={i:t for i,t in  enumerate(config["entity_type"])}


config["entity1_relation_type"]=['Participant', 'Agent', 'Domain', 'Molecule', 'Element1', 'Functional_Molecule', 'Element', 'Source', 'Amino_Acid_Sequence', 'DNA_Part', 'Process', 'Agent1']
config["entity1_relation_type2id"]={t:i for i,t in  enumerate(config["entity_relation_type"])}
config["id2entity1_relation_type"]={i:t for i,t in  enumerate(config["entity_relation_type"])}


config["entity2_relation_type"]=['Element2', 'Product', 'Development', 'DNA', 'Agent2', 'Target',
                                 'Genotype', 'Molecule', 'Family', 'Target_Tissue', 'Functional_Molecule', 'Protein_Complex', 'Process']
config["entity2_relation_type2id"]={t:i for i,t in  enumerate(config["entity_relation_type"])}
config["id2entity2_relation_type"]={i:t for i,t in  enumerate(config["entity_relation_type"])}





config["relation_type"]=['Is_Involved_In_Process', 'Regulates_Expression', 'Has_Sequence_Identical_To', 'Exists_In_Genotype',
                         'Transcribes_Or_Translates_To', 'Occurs_In_Genotype', 'Composes_Protein_Complex', 'Is_Linked_To',
                         'Regulates_Process', 'Interacts_With', 'Binds_To', 'Occurs_During', 'Is_Member_Of_Family',
                         'Is_Functionally_Equivalent_To',
                         'Is_Localized_In', 'Regulates_Accumulation',
                          'Composes_Primary_Structure', 'Exists_At_Stage', 'Regulates_Development_Phase',
                         'Is_Protein_Domain_Of', 'Regulates_Tissue_Development', 'Regulates_Molecule_Activity']
config["relation_type2id"]={t:i for i,t in  enumerate(config["relation_type"])}
config["id2relation_type"]={i:t for i,t in  enumerate(config["relation_type"])}


config["max_seq_len"]=156 #
config["max_position_len"]=156 #


config["max_different_position_len"]=78
config["position_dim"]=10
config["batch_size"]=64

