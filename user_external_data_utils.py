#Import libraries
from user_data_utils import clean_database, process_database, fillna_database,preprocess_filtered_database
import pandas as pd

def load_external_database():
	# df = pd.read_excel("External Database.xlsx")
	return df

def clean_external_database(df):
	df = clean_database(df)
	return df

def process_external_database(df):
	df = process_database(df)
	return df

def fillna_external_database(df):
	df=fillna_database(df)
	return df

def preprocess_filtered_external_database(df, wf_name):
	df=preprocess_filtered_database(df, wf_name)
	return df
