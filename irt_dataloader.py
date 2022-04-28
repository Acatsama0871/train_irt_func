import os
import ast
import json
import copy
import random
import warnings
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class IRTDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.dataset = dataset
        self.data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle).__iter__()
        print(len(self.dataset))

    def next_batch(self):
        try:
            return next(self.data_loader)
        except StopIteration:
            self.data_loader = DataLoader(self.dataset, self.batch_size, shuffle=self.shuffle).__iter__()
            return self.next_batch()

# items_df is a dataframe with these columns:
#ID, Pos_support_locs	Neg_support_locs	Query_loc	Label	Alpha	Aug_type	diff

class IRTDataset(Dataset):
    def __init__(self, items_df):
        self.items_df = items_df

    def __getitem__(self, index):
        
        item = self.items_df.iloc[index].to_dict()
        
        return item

    def __len__(self):
        return len(self.items_df)


class CL_dff_loader:
    
    def __init__(self, data_df, batch_size):
        # attributes
        self.batch_size = batch_size
        self.data_df = data_df
        

    def batch_generator_with_theta(self, theta):
        # find (diff < theta) ids
        id_subset = self.data_df[self.data_df["diff"] <= theta]

        if len(id_subset) < self.batch_size:
            raise NoSample
        # get dataset and dataloader
        irt_dataset = IRTDataset(id_subset)

        return IRTDataLoader(irt_dataset, batch_size = self.batch_size, shuffle=True)


    def batch_generator_without_theta(self, subsample_size=-1):
        if subsample_size == -1:
            # get dataset and dataloader
            irt_dataset = IRTDataset(self.data_df)
        else:
            irt_dataset = IRTDataset(self.data_df.sample(subsample_size))
        
        return IRTDataLoader(irt_dataset, batch_size = self.batch_size, shuffle=True)


class NoSample(Exception):
    def __init__(self):
        self.message = "No sample's diff is lower than current theta"
        super().__init__(self.message)


class OnlyOneSample(Exception):
    def __init__(self):
        self.message = "Only one sample is found, minimum requirement: 2"
        super().__init__(self.message)