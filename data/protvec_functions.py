import pandas as pd
from functions import init_code, tuple_str
from textwrap import wrap
import random

#######

def three_embedding_dict():

	df = pd.read_csv('/Users/0/Desktop/2strucpred/dict_files/103_embedding.csv', header=None)

	#convert the df rows into a list of rows
	rows_as_list=[]
	for row in df.iterrows():
	    index, data = row
	    rows_as_list.append(data.tolist())


	# make protvec dictionary
	protvec_dict = {}
	for value in rows_as_list:
		three_letter = value[0]
		#print (three_letter)
		dimension = value[1:105]
		protvec_dict[three_letter] = dimension

	return (protvec_dict)

def eight_embedding_dict():

	df = pd.read_csv('/Users/0/Desktop/2strucpred/dict_files/108_embedding.csv', header=None)

	#convert the df rows into a list of rows
	rows_as_list=[]
	for row in df.iterrows():
	    index, data = row
	    rows_as_list.append(data.tolist())


	# make protvec dictionary
	protvec_dict = {}
	for value in rows_as_list:
		three_letter = value[0]
		#print (three_letter)
		dimension = value[1:109]
		protvec_dict[three_letter] = dimension

	return (protvec_dict)

