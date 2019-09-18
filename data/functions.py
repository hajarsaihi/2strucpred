#######################################################################################################################
###### IMPORT TOOLS
import pandas as pd
import numpy as np
from numpy import array
import itertools
from itertools import islice
import os

#######################################################################################################################
###### FUNCTIONS
def init_chain(textfile): # this make a df with the PDB IDs and what chain for each one to use
	path = "/Volumes/Hajar's HDD/MSc_data/large_proteins/PDB_files/PDB_with_chain/"+textfile
	chainli = open(path, 'r') # open the chain list file
	pdb_df = pd.read_csv(chainli, names = ['PDB_ID']) # read this into a pandas df
	pdb_df = pdb_df.drop_duplicates() # some PDB IDs are repeated
	pdb_df['chain'] = pdb_df['PDB_ID'].str[4:5] # only keep the first letter of the chain
	pdb_df['PDB_ID'] = pdb_df['PDB_ID'].str[:4] # def the PDB ID
	return pdb_df 

def dssp_to_df(dssp_file, chain_file):
	'''convert the dssp file into a df for easy use later'''
	path = "/Volumes/Hajar's HDD/MSc_data/large_proteins/PDB_files/validation_PDB/"+dssp_file # def the path loc dep on pdb set (test/val/train)

	df = pd.read_csv(path, names = ['Residue','Chain','Position','AA','SS','3H','4H','5H','Bend','Chiral',
		'BB1','BB2', 'BBres','BSlabel', 'Solvent_Acc']) # import dssp file into a pandas df
	df = df.drop(columns=['3H','4H','5H','Bend','Chiral','BB1','BB2', 'BBres','BSlabel', 'Solvent_Acc']) # dont need these, drop them

	# clean the data:
	df['Residue'] = df['Residue'].astype(str).str.replace(r"(", '') # remove the brackets
	#df['Solvent_Acc'] = df['Solvent_Acc'].astype(str).str.replace(r")", '')
	df['AA'] = df['AA'].astype(str).str.replace(r"'", '')
	df['SS'] = df['SS'].astype(str).str.replace(r"'", '')
	df['Chain'] = df['Chain'].astype(str).str.replace(r"'", '') # remove the '
	df['Chain'] = df['Chain'].astype(str).str.replace(r" ", '')

	pdb_df = init_chain(chain_file) # given the chain file, only use the df with that specific chain mentioned by ProteinNet

	ID = dssp_file[0:4]#.lower() # for test set use lower() because of the filenames

	if ID in pdb_df.PDB_ID.values: 
		chain = pdb_df.loc[pdb_df['PDB_ID'] == ID]['chain'] # check if the PDB IDs are a match
		chain = chain.values[0] # get the chain value
		df = df[df.Chain.str.contains(chain)] # only keep the rows that have that specific chain
	else:
		print (ID, 'please try again')
	df = df[df.AA != ' X'] # drop the rows with undetermined amino acids
  
	return df


def get_all_sequence(df, col): 
	'''get the sequence information'''
	seq = [] # list of the residues
	for x in df[col]:
		x = str(x).strip()
		seq.append(x)
	my_lst_str = ''.join(map(str, seq)).strip() # use this if you just want the seq not in a list.
	return (my_lst_str)

def get_all_chain(df): 
	'''get chain information from the dataframe'''
	chain = [] # list of the residues
	for x in df['Chain']:
		x = str(x).strip()
		chain.append(x)
	return chain

def all_window(seq, n): 
	'''generate the sliding windows by iterating over the sequence'''
	it = iter(seq) # iterate over the sequence
	result = tuple(islice(it, n)) # slice based on iteration and n - sliding window size
	som = []

	if len(result) == n:
		som.append(result)
	for elem in it:
		result = result[1:] + (elem,)
		som.append(result)
	return som

def tuple_str(window): 
	'''For the one hot encoding, the str version of the seq is needed, not the tuple.'''
	string_list = []
	for value in window:
		aa_str =''.join(value)
		string_list.append(aa_str)
	return string_list

def init_code (dssp_file):
	'''start up code combining the functions to yield a string of amino acids'''
	df = dssp_to_df (dssp_file, 'val_IDs_chain.txt')
	# AA
  seq = get_all_sequence(df, 'AA')
  seq_list = list(seq)
  
  # SS
	ss = get_all_sequence(df, 'SS')
  ss_list = eight_to_3(ss_list) # !convert the eight structures into three - if not, hash this out!
	ss_list = list(ss)

	aa_string = tuple_str(seq_list) # convert them into string
	return seq_list, ss_list

def one_hot_encode (categories, data):
	'''apply one hot encoding given the categories and the data'''
	amino_acids = categories # this is all the categories including X for none

	# define a mapping of chars to integers
	char_to_int = dict((c, i) for i, c in enumerate(amino_acids))
	int_to_char = dict((i, c) for i, c in enumerate(amino_acids))

	#data = x # this is the window of letters
	integer_encoded = [char_to_int[char] for char in data]

	onehot_encoded = list()
	for value in integer_encoded:
		letter = [0 for _ in range(len(amino_acids))]
		letter[value] = 1
		onehot_encoded.append(letter)
	return onehot_encoded

def eight_to_3(ss_string): 
  '''convert 8 class ss into 3 class ss'''
	ss_str = []
	for x in ss_string:
		if x == 'T' or x == 'S' or x == 'I' or x == 'G' or x == 'B' or x == '-':
			x = 'C'
			ss_str.append(x)
		else:
			ss_str.append(x)
	return ss_str
