'''
data helper to preprocess csv format text dataset

Jingfeng Zhang (gmail: jingfeng.zhang9660@gmail.com) 
'''
import csv
import numpy as np

class data_helper():
	def __init__(self, sequence_max_length=1024, noise_level = 0.1, seed = 7, mode = "train"):
		seed_num = seed
		np.random.seed(seed_num)
		self.alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} '+'\\'
		self.char_dict = {}
		self.sequence_max_length = sequence_max_length
		self.noise_level = noise_level
		self.mode = mode
		for i,c in enumerate(self.alphabet):
			self.char_dict[c] = i+1

	def char2vec(self, text):
		data = np.zeros(self.sequence_max_length)
        
		if len(text) < self.sequence_max_length:
			len_text = len(text)
		else:
			len_text = self.sequence_max_length
		s1 = np.random.randint(len_text, size=int(self.noise_level * len_text ) )   ## randomly choose positions. 
		noise = np.random.randint(26, size=int(self.noise_level * len_text ))  + 1  # randomly change to a - z 
		for i in range(0, len_text):
			#if i >= self.sequence_max_length: # This should be >= 
				#return data
			if text[i] in self.char_dict:
				data[i] = self.char_dict[text[i]]
			else:
				# unknown character set to be 69
				data[i] = 69
		if self.mode == "train":
			data[s1] = noise  ## add noise to chars. 
		return data

	def load_csv_file(self, filename, num_classes):
		"""
		Load CSV file, generate one-hot labels and process text data as Paper did.
		"""
		all_data = []
		labels = []
		with open(filename) as f:
			reader = csv.DictReader(f, fieldnames=['class'], restkey='fields')
			for row in reader:
				# One-hot
				#one_hot = np.zeros(num_classes)
				#one_hot[int(row['class']) - 1] = 1
				labels.append(int(row['class']) -1 )
				# we are using title and description for classifition, if not use title, specify a_text = row['field][1:]
				text = ''
				a_text = row['fields']
				for i in a_text:
					text +=i.lower()
					text +=' '
				all_data.append(self.char2vec(text))
		f.close()
		return np.array(all_data, dtype=int), np.array(labels, dtype=int)

	def load_dataset(self, dataset_path):
		# Read Classes Info
		with open(dataset_path+"classes.txt") as f:
			classes = []
			for line in f:
				classes.append(line.strip())
		f.close()
		num_classes = len(classes)
		# Read CSV Info
		train_data, train_label = self.load_csv_file(dataset_path+'train.csv', num_classes)
		self.mode = "test" 
		test_data, test_label = self.load_csv_file(dataset_path+'test.csv', num_classes)
		return train_data, train_label, test_data, test_label

