from .davis import *


test_results_dict = {}
with open('results_dict.pickle', 'rb') as fp:
	test_results_dict = pickle.load(fp)
	
train_results_dict = {}
with open('train_results_dict.pickle', 'rb') as fp:
	train_results_dict = pickle.load(fp)
	
val_results_dict = {}
with open('val_results_dict.pickle', 'rb') as fp:
	val_results_dict = pickle.load(fp)


