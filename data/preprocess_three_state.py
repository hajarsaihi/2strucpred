##### (1) Import Modules ###################################################################################################
from itertools import islice
from functions import tuple_str, all_window, init_code
import protvec_dict, factor_of_three, eight_embedding_dict, new_embedding_dict
from numpy import array
import numpy as np
import glob

##### (2) Get Data #########################################################################################################
filenames = glob.glob("/Volumes/Hajar's HDD/MSc_data/large_proteins/PDB_files/train30_PDB/set1196/*.dssp") # set the location of DSSP files
PDB_files = []

for path in filenames:
	file = path[-9:] # last 9 is dssp file name
	PDB_files.append(file) # create a list of just the file names e.g. XXX.dssp

##### (3) Def Dictionary and lists #########################################################################################
aa_array = []
error_files = []
counter = 0

##### (4) Run Main Code ####################################################################################################
#### Depending on either eight or three state alternate between which dictionary to use!
threeGrams = new_embedding_dict() # import the dictionary for three state (103 dimensions)
#threeGrams = eight_embedding_dict() # import the dictionary for eight state (108 dimensions)
empty_word = [0] * 103 # padding vector - length dep on 103 or 108 

for file in PDB_files: # iterate over 'PDB_files' list with dssp names
	counter = counter + 1
	print (counter, 'out of', len(PDB_files))
	try:
		seq, ss = init_code(file) # run the initiliase code script to get the sequence and secondary structure
		final_list = ['X'] * 8 # add padding value to start
		seq.extend(final_list) # add padding value to end
		final_list.extend(seq)

		grams = all_window(final_list, 17) # get windows of 17 amino acids
		grams = tuple_str(grams) # convert to string, 'grams' is a list of the windows

		for window in grams:
			word_list = all_window(window, 3) # for each window length 17 aa extract the words
			word_list = tuple_str(word_list) # convert tuple to str
			window_vec = []
			for word in word_list:
				if 'X' in word:
					prot_vec_3_gram = empty_word # if the word has the padding value, it is represented by the empty vector...
					window_vec.append(prot_vec_3_gram)
					
				elif ('X' not in word) and (word in threeGrams.keys()): #... if not, and the word is in the dictionary, it is represented by the embedding in the dictionary
					prot_vec_3_gram = threeGrams[word] # get the triplet vector from the 103/108 dim dictionary
					window_vec.append(prot_vec_3_gram) # append it instead of extending

			aa_array.append(window_vec) # append vector of each window to yield final array
			
				
	except ValueError: # handle any exceptions
		error_files.append(file)
		print ('ERROR', file) # some files dont have the chain thats mentioned in the main file
		continue

a = array(aa_array)
print (a.shape)

##### (5) Output File ####################################################################################################
np.save('x_train.npy', a, allow_pickle=True) # save the file - change name dep on dim
