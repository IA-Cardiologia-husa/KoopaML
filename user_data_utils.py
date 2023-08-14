#Import libraries
import pandas as pd

def load_database():
	# df = pd.read_excel("Database.xlsx")
	return df

def clean_database(df):
	return df

def process_database(df, wf_name):
	return df

def load_external_database():
	# df = pd.read_excel("External Database.xlsx")
	return df

def clean_external_database(df):
	df = clean_database(df)
	return df

def process_external_database(df, wf_name):
	df = process_database(df, wf_name)
	return df
