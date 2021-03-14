import sys # for testing

import numpy as np
import pandas as pd

import torch
import torch.tensor as ts
from torch.utils.data import Dataset

import gen_demo_data

'''
Data handler - read, generate and manage the dataset.
    1. Use demo data by defining 'func' (in 'gen_demo_data');
    2. Use imported data by defining 'dir' (the data directory).
'''

class Data_Handler():
    
    def __init__(self, file_dir=None, func=None, val_proportion=0.2, batch_size=1, is_shuffle=True):
        super().__init__()
        assert((func is not None) | (dir is not None)), ("Define 'func' or 'dir'.")
        print("Generating dataset...")
        if file_dir is None:
            self.__data, self.__labels, self.__mean = func
            self.data_shape = self.__data.shape
            self.label_shape = self.__labels.shape
            self.__num_data = self.data_shape[0]
        else:
            df = pd.read_csv(file_dir)
            self.__data = df.to_numpy()[:,:-2]
            self.__data[:,:-2] = self.__data[:,:-2]
            self.__labels = df.to_numpy()[:,-2:]
            self.data_shape = self.__data.shape
            self.label_shape = self.__labels.shape
            self.__num_data = self.data_shape[0]
        self.val_p = val_proportion
        self.batch_size = batch_size
        self.__batch_pointer = 0
        self.split_train_val()
        if is_shuffle:
            self.shuffle()


    def split_train_val(self, val_limit=int(1e4)):
        n_val = int(self.val_p*self.__num_data)
        self.__data_val = self.__data[:n_val,:]
        self.__data_train = self.__data[n_val:,:]
        self.__labels_val = self.__labels[:n_val,:]
        self.__labels_train = self.__labels[n_val:,:]
        return self.__data_val[:val_limit,:], self.__labels_val[:val_limit,:], self.__data_train, self.__labels_train

    def return_num_data(self):
        return self.__num_data

    def return_data(self):
        return self.__data, self.__labels, self.__mean

    def return_dataset(self):
        return list(zip(self.__data_train, self.__labels_train))

    def return_dataset_ts(self):
        return list(zip(ts(self.__data_train),ts(self.__labels_train)))

    def return_batch(self, batch_size=None):
        data = self.__data_train
        labels = self.__labels_train
        bs = batch_size
        if bs is None:
            bs = self.batch_size
        bp = self.__batch_pointer
        if bs+bp <= data.shape[0]:
            batch = ts(data[bp:bp+bs,:])
            batch_label = ts(labels[bp:bp+bs,:])
            self.__batch_pointer += bs
        else:
            batch = ts(np.concatenate((data[bp:,:],data[:bs+bp-data.shape[0],:]),axis=0))
            self.__batch_pointer = bs+bp-data.shape[0]
            batch_label = ts(np.concatenate((labels[bp:,:],labels[:bs+bp-data.shape[0],:]),axis=0))
        if len(np.shape(batch)) < len(np.shape(data)):
            minibatch = minibatch[np.newaxis, :] # add batch size
        if len(np.shape(batch_label)) < len(np.shape(labels)):
            label = label[np.newaxis, :] # add batch size
        return batch.float(), batch_label.float()

    def set_batch_pointer(self, x=0):
        self.__batch_pointer = x

    def get_batch_pointer(self):
        return self.__batch_pointer

    def shuffle(self):
        dataset_ts = self.return_dataset_ts()
        np.random.shuffle(dataset_ts)
        x, y = zip(*dataset_ts)
        self.__data   = np.array([i.tolist() for i in x])
        self.__labels = np.array([i.tolist() for i in y])
        print('Shuffled.')


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    file = '210210_p3m10_2000s_246_249_1414.csv'

    print("Test data handler.")
    # maxT = 10
    # past = 1
    # func = gen_demo_data.gen_DemoBeta(nsamples=40, maxT=maxT, past=past, vis=0)
    DH = Data_Handler(file_dir=file)
    batch,label = DH.return_batch(1)
    print(batch, label)
    fig, ax = plt.subplots()
    gen_demo_data.load_oneMap(ax)
    plt.plot(label[0,0],label[0,1],'rx')
    plt.show()


    