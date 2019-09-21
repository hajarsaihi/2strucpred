##### (1) Import Modules ###################################################################################################
from functions import all_window, tuple_str, init_code, one_hot_encode
import glob
import itertools
from itertools import islice, repeat
from numpy import array
import numpy as np

##### (2) Get Data #########################################################################################################
filenames = glob.glob("/Users/0/Desktop/2strucpred/data/dssp_files/train/*.dssp")
error_files = []
PDB_files = []
#PDB_files = ['1A0A.dssp', '1J1J.dssp', '1J2G.dssp'] 

for path in filenames:
	file = path[-9:] # last 9 is dssp file name
	PDB_files.append(file)

##### (3) Def Dictionary and lists #########################################################################################
ss_array = []
count = 0 

##### (4) Run Main Code ####################################################################################################
for file in PDB_files:
	try:
		count = count+1
		print ('Position:', count, 'out of', len(PDB_files))
		seq, ss = init_code(file) # run the initiliase code script to get the sequence and secondary structure
		ss_string = ss

		for data in ss_string:
			onehot_encoded = one_hot_encode('BEGHIST-', data) # Use this for eight state
			#onehot_encoded = one_hot_encode('EHC', data) # Use this for three state
			final_one_hot = np.array(onehot_encoded)
			final_one_hot = final_one_hot.flatten() # flatten the data
			y = final_one_hot.tolist()
			ss_array.append(y) # append one hot encoded structure to array

	except ValueError: # Handle exceptions
		error_files.append(file)
		print ('ERROR', file) # some files dont have the chain thats mentioned in the main file
		continue

a = array(ss_array)
print (a.shape)

##### (5) Output File ####################################################################################################
np.save('y_train.npy', a, allow_pickle=True) # save the file - change name dep on train/val sets

