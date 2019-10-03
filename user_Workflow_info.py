# In this archive we have to define the dictionary WF_info. This is a dictionary of dictionaries, that for each of our workflows
# assigns a dictionary that contains:
#
# formal_title:		Title for plots and reports
# label_name:		Variable to predict in this workflow, e.g.: 'Var17'
# feature_list:		List of features to use in the ML models, e.g.: ['Var1', 'Var2', 'Var4']
# filter_function:	Function to filter the Dataframe. If we want to use only the subject of the dataframe with Var3=1, 
# 					we would write: lambda df: df.loc[df['Var3']==1].reset_index(drop=True)
# 					In case we want no filter, we have to write: lambda df: df
#
# Example:
# 
# WF_info['TallHeart'] = {'formal_title': 'Prediction of Heart Attack in tall patients',
#						  'label_name': 'Heart Attack',
#						  'feature_list': ['Age','Height','Weight','Arterial Tension'],
#						  'filter_funtion': lambda df: df.loc[df['Height']>200].reset_index(drop=True)


WF_info ={}