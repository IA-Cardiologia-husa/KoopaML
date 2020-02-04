# In this archive we have to define the dictionary RS_info. This is a dictionary of dictionaries, that for each of the Risk Scores
# assigns a dictionary that contains:
#
# label_name: the name of the variable of the dataframe that contains the value of the risk score
# formal_name: name to be used in plots and report
#
# Example:
# RS_info = {}
#
#RS_info['CHADSVASC'] = {'label_name':'CHADSVASC',
#						 'formal_name': 'CHA2DS2-VASc'
# 						 'sign': -1}

# RS_info['CHADSVASC'] = {'feature_oddratio':{'ICC':1,
# 											'HTA':1,
# 											'age75':1,
# 											'DM':1,
# 											'ICTUS/AIT PREVIO':1,
# 											'Age65':1,
# 											'Genero'-1:
# 											},
# 						'formal_name': 'CHA2DS2-VASc',
# 						'refit_oddratio':'No'}

RS_info = {}
