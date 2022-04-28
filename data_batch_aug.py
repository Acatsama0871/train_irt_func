from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import glob
import torch

import scipy.misc
import numpy as np
from scipy.special import comb

from tqdm import tqdm
import torch

            
class CL_Data_loader():
    
    # data is a list with all feature arrays 
    # X_train_pos, X_train_neg, X_val_pos,X_val_neg only contain indexes for train and validation
    # augmentation degree: alpha [0, 0.1 - 0.5], 
    # augmention method [1-4], insert, swap, delete, replace
    # No Augmentation: alpha 0, method 0 
    # data: original data
    # aug_data: augmented data in the format of dictionary, i.e. aug_data[alpha][method]
    
    def __init__(self, X_train_pos, X_train_neg, X_val_pos,X_val_neg, \
                 data, aug_data, batch_size, k_shot=1, train_mode=True):  
        
        self.data = data
        self.aug_data = aug_data
        
        self.batch_size = batch_size
        
        self.k_shot = k_shot # 1 or 5, how many times the model sees the example
        
        self.num_classes = 2   # this is a binary classification
        
        self.train_pos = X_train_pos
        self.train_neg = X_train_neg

        # position of last batch
        self.train_pos_index = 0
        self.train_neg_index = 0
            
        if not train_mode:
            
            self.val_pos = X_val_pos
            self.val_neg = X_val_neg
               
            self.val_pos_index = 0
            self.val_neg_index = 0
                  
            # merge train & val for prediction use
            self.all_pos = np.concatenate([self.train_pos, self.val_pos])
            self.all_neg = np.concatenate([self.train_neg, self.val_neg])

            self.pos_index = 0
            self.neg_index = 0
            
        
        self.iters = 100


    def next_batch(self, alpha=0, aug_type = 0, return_sample_ids=False):  
        x_set_batch = []
        y_set_batch = []
        x_hat_batch = []
        y_hat_batch = []
        
        
        x_set = []
        y_set = []
        
        for _ in range(self.batch_size):
            
            x_set = []
            y_set = []
        
            target_class = np.random.randint(self.num_classes)
            #print(target_class)                
                    
            # negative class
            for i in range(self.k_shot+1):
                
               # shuffle pos or neg if a sequence has been full used
                if self.train_neg_index == len(self.train_neg):
                    
                    self.train_neg = np.random.permutation(self.train_neg)
                    self.train_neg_index = 0
                    #print("neg seq", self.train_neg_seq)
                    
                if i==self.k_shot:  # the last one is test sample
                    
                    if target_class == 0: # positive class
                        x_hat_batch.append(self.train_neg[self.train_neg_index])
                        
                        y_hat_batch.append(0)
                        self.train_neg_index += 1                    
                else:
                    
                    x_set.append(self.train_neg[self.train_neg_index])
                    
                    y_set.append(0)
                    self.train_neg_index += 1
                                       
             # positive class
            for i in range(self.k_shot+1):
                
               # shuffle pos or neg if a sequence has been full used
                if self.train_pos_index == len(self.train_pos):
                    
                    self.train_pos = np.random.permutation(self.train_pos)
                    self.train_pos_index = 0
                    #print("pos seq", self.train_pos_seq)
                    
                if i==self.k_shot:  # the last one is test sample
                    
                    if target_class == 1: # positive class
                        x_hat_batch.append(self.train_pos[self.train_pos_index])
                        
                        y_hat_batch.append(1)
                        self.train_pos_index += 1
                
                        
                else:
                    x_set.append(self.train_pos[self.train_pos_index])
                    
                    y_set.append(1)
                    self.train_pos_index += 1

            x_set_batch.append(x_set)
            y_set_batch.append(y_set)       
        
        # get feature arrays for the batch
        
        #print(x_set_batch)
        #print(x_hat_batch)
        
        feature_set_batch = []
        feature_hat_batch = []
        
        for did, feature in enumerate(self.data):
            if did == 0:    # word vector
                
                if alpha == 0:  # no augmentation
                    f_set = np.array([np.array(feature[b]) for b in x_set_batch])
                    f_hat = np.array(feature[x_hat_batch])
                else:
                    feature = self.aug_data[alpha-1][aug_type -1]
                    f_set = np.array([np.array(feature[b]) for b in x_set_batch])
                    f_hat = np.array(feature[x_hat_batch])
                    
            else:
                f_set = np.array([np.array(feature[b]) for b in x_set_batch])
                f_hat = np.array(feature[x_hat_batch])

            # reshape support to (batch, n_way, k_shot, *feature size)
            f_set = f_set.reshape((self.batch_size, 2, self.k_shot, *(feature.shape[1:])))
            #print(f_set.shape)
            #print(f_hat.shape)
            
            feature_set_batch.append(f_set)
            feature_hat_batch.append(f_hat)
            
        feature_set_batch = self.convert_to_tensor(feature_set_batch)
        feature_hat_batch = self.convert_to_tensor(feature_hat_batch)
        y_hat_batch = torch.Tensor(np.asarray(y_hat_batch).astype(np.int32))
            
        if return_sample_ids:           
            return feature_set_batch, feature_hat_batch, y_hat_batch, \
                   x_set_batch, x_hat_batch  # sample IDs 
        else:
            return feature_set_batch, feature_hat_batch, y_hat_batch
               #np.zeros(self.batch_size)   # all 0s for aux output


    def next_eval_batch(self, return_sample_ids = False):
        x_set_batch = []
        y_set_batch = []
        x_hat_batch = []
        y_hat_batch = []
        
        for _ in range(self.batch_size):
            
            x_set = []
            y_set = []
            
            target_class = np.random.randint(self.num_classes)
            #print(target_class)
            
            if self.val_pos_index == len(self.val_pos):
                self.val_pos = np.random.permutation(self.val_pos)
                self.val_pos_index = 0
                #print("pos val seq", self.val_pos_seq)

            if self.val_neg_index == len(self.val_neg):
                self.val_neg = np.random.permutation(self.val_neg)
                self.val_neg_index = 0
                #print("net val seq", self.val_neg_seq)
                           
            # negative class
            for i in range(self.k_shot+1):
                
               # shuffle pos or neg if a sequence has been full used
                if self.train_neg_index == len(self.train_neg):
                    self.train_neg = np.random.permutation(self.train_neg)
                    self.train_neg_index = 0
                    #print("neg seq", self.train_neg_seq)
                    
                if i==self.k_shot:  # the last one is test sample
                    
                    if target_class == 0: # negative class
                        x_hat_batch.append(self.val_neg[self.val_neg_index])
                        
                        y_hat_batch.append(0)
                        self.val_neg_index += 1       
                else:
                    
                    x_set.append(self.train_neg[self.train_neg_index])
                    
                    y_set.append(0)
                    self.train_neg_index += 1

            # positive class
            for i in range(self.k_shot+1):
                
               # shuffle pos or neg if a sequence has been full used
                if self.train_pos_index == len(self.train_pos):
                    self.train_pos = np.random.permutation(self.train_pos)
                    self.train_pos_index = 0
                    #print("pos seq", self.train_pos_seq)
                    
                if i==self.k_shot:  # the last one is test sample
                    
                    if target_class == 1: # positive class
                        x_hat_batch.append(self.val_pos[self.val_pos_index])
                        
                        y_hat_batch.append(1)
                        self.val_pos_index += 1               
                        
                else:
                    x_set.append(self.train_pos[self.train_pos_index])
                    
                    y_set.append(1)
                    self.train_pos_index += 1

                      
            x_set_batch.append(x_set)
            y_set_batch.append(y_set)
        
        #print(x_set_batch)
        #print(x_hat_batch)
        
        feature_set_batch = []
        feature_hat_batch = []
        
        # loop through all features 
        for feature in self.data:
            f_set = np.array([np.array(feature[b]) for b in x_set_batch])
            f_hat = np.array(feature[x_hat_batch])
            #print(f_set.shape)
            #print(f_hat.shape)

            f_set = f_set.reshape((self.batch_size, 2, self.k_shot, *(feature.shape[1:])))
            
            feature_set_batch.append(f_set)
            feature_hat_batch.append(f_hat)
        
            
        if return_sample_ids:           
            return feature_set_batch, np.asarray(y_set_batch).astype(np.int32), \
               feature_hat_batch, np.asarray(y_hat_batch).astype(np.int32),\
               x_set_batch, x_hat_batch  # get sample IDs
        else:
            return feature_set_batch, np.asarray(y_set_batch).astype(np.int32), \
               feature_hat_batch, np.asarray(y_hat_batch).astype(np.int32)
               
   
                
    
    # generate support set for each sample in prediction
    # use all samples as support
    def get_pred_set(self, pred, return_sample_ids=False):
        x_set_batch = []
        y_set_batch = []
        x_hat_batch = []
        
        for _ in range(self.batch_size):  #batch_size = 32
            
            x_set = []
            y_set = []
            
            target_class = np.random.randint(self.num_classes)   #target_class = 0/1
            #print(target_class)
            
            if self.pos_index == len(self.all_pos):  #initiate the index
                self.all_pos = np.random.permutation(self.all_pos)
                self.pos_index = 0

            if self.neg_index == len(self.all_neg):   #initiate the index
                self.all_neg = np.random.permutation(self.all_neg)
                self.neg_index = 0
                           
            # negative class
            for i in range(self.k_shot):
                
               # shuffle pos or neg if a sequence has been full used
                if self.neg_index == len(self.all_neg):
                    self.all_neg = np.random.permutation(self.all_neg)
                    self.neg_index = 0
                    #print("neg seq", self.train_neg_seq)
                    
                x_set.append(self.all_neg[self.neg_index])
                
                y_set.append(0)
                self.neg_index += 1

            # positive class
            for i in range(self.k_shot):
                
               # shuffle pos or neg if a sequence has been full used
                if self.pos_index == len(self.all_pos):
                    self.all_pos = np.random.permutation(self.all_pos)
                    self.pos_index = 0
                    #print("pos seq", self.train_pos_seq)

                x_set.append(self.all_pos[self.pos_index])
                
                y_set.append(1)
                self.pos_index += 1
                    
            # Prediction sample
            
            x_set_batch.append(x_set)
            y_set_batch.append(y_set) 
        
        x_hat_batch.append(pred)
        #print(x_set_batch)

        # repeat each element in pred for batch_size times
        feature_hat_batch = [np.repeat(e[None,:], self.batch_size, axis = 0) for e in pred]
        feature_set_batch = []
        
        # loop through all features 
        for idx, feature in enumerate(self.data):
            f_set = np.array([np.array(feature[b]) for b in x_set_batch])
            #print(f_set.shape)

            f_set = f_set.reshape((self.batch_size, 2, self.k_shot, *(feature.shape[1:])))
            
            feature_set_batch.append(f_set)
            
        if return_sample_ids:
            return feature_set_batch, np.asarray(y_set_batch).astype(np.int32), feature_hat_batch, x_set_batch
        else:           
            return feature_set_batch, np.asarray(y_set_batch).astype(np.int32), feature_hat_batch
    
    #def get_pred_set_gen(self, pred):
    #    while True:
    #        x_set, y_set, x_hat, y_hat = train_loader.next_batch()
    #        yield([x_set, x_hat], 1-y_hat)

    def convert_to_tensor(self, features):
      tensors = [torch.Tensor(item) for item in features]
      return tensors

            
    def next_eval_batch_gen(self, return_sample_ids=False):
        while True:
            if return_sample_ids:
                x_set, y_set, x_hat, y_hat, x_set_ids, x_hat_ids = self.next_eval_batch(return_sample_ids = return_sample_ids)
            else:
                x_set, y_set, x_hat, y_hat = self.next_eval_batch(return_sample_ids = return_sample_ids)
            
            x_set = self.convert_to_tensor(x_set)
            x_hat = self.convert_to_tensor(x_hat)
            y_hat = torch.Tensor(y_hat)
            
            if return_sample_ids:
                yield (x_set, x_hat, y_hat, x_set_ids, x_hat_ids)
            else:
                yield(x_set, x_hat, y_hat)
            
    
    
    def next_batch_by_ids(self, x_set_ids, x_hat_ids, y_set, y_hat):
        feature_set_batch = []
        feature_hat_batch = []
        
        # loop through all features 
        for feature in self.data:
            f_set = np.array([np.array(feature[b]) for b in x_set_ids])
            f_hat = np.array(feature[x_hat_ids])
            #print(f_set.shape)
            #print(f_hat.shape)

            f_set = f_set.reshape((self.batch_size, 2, self.k_shot, *(feature.shape[1:])))
            
            feature_set_batch.append(f_set)
            feature_hat_batch.append(f_hat)
        
            return (feature_set_batch, y_set, feature_hat_batch, y_hat)
        
        
    # irt_ # items is a dictionary with keys:
    #ID, Pos_support_locs	Neg_support_locs	Query_loc	Label	Alpha	Aug_type	dif
    def next_batch_by_irt_items(self, irt_items):
        
        feature_set_batch = []
        feature_hat_batch = []
        y_hat_batch = []
        #y_set_batch = []      
        
        neg_support = torch.stack(irt_items["Neg_support_locs"], dim = 1).numpy()  # batch_size x k_shot
        pos_support = torch.stack(irt_items["Pos_support_locs"], dim = 1).numpy()
        x_hat_batch = irt_items["Query_loc"].numpy()
        alpha_batch = irt_items["Alpha"].numpy()
        aug_type_batch = irt_items["Aug_type"].numpy()
        y_hat_batch = irt_items["Label"].numpy()
        
        #print(neg_support.shape, pos_support.shape)    
        
        x_set_batch  = np.concatenate([neg_support, pos_support], axis = 1)
        
        for did, feature in enumerate(self.data):
            
            f_set = []
            f_hat = []
            
            for i in range(len(x_set_batch)):
                    
                x_set = x_set_batch[i]
                x_hat = x_hat_batch[i]
                alpha = alpha_batch[i]
                aug_type = aug_type_batch[i]
            
                if did == 0:    # word vector

                    if alpha == 0:  # no augmentation
                        f_set += [np.array(feature[b]) for b in x_set]
                        f_hat.append(feature[x_hat])
                    else:
                        feature = self.aug_data[alpha-1][aug_type -1]
                        f_set += [np.array(feature[b]) for b in x_set]
                        f_hat.append(feature[x_hat])

                else:
                    f_set += [np.array(feature[b]) for b in x_set]
                    f_hat.append(feature[x_hat])

            f_set = np.array(f_set)
            f_hat = np.array(f_hat)

            # reshape support to (batch, n_way, k_shot, *feature size)
            f_set = f_set.reshape((-1, 2, self.k_shot, *(feature.shape[1:])))
            #print(f_set.shape)
            #print(f_hat.shape)

            feature_set_batch.append(f_set)
            feature_hat_batch.append(f_hat)
            
        feature_set_batch = self.convert_to_tensor(feature_set_batch)
        feature_hat_batch = self.convert_to_tensor(feature_hat_batch)
        y_hat_batch = torch.Tensor(np.asarray(y_hat_batch).astype(np.int32))
        #y_set_batch = torch.Tensor(np.asarray(y_set_batch).astype(np.int32))
        
               
        return (feature_set_batch, feature_hat_batch, y_hat_batch)
            
 