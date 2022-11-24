# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:40:50 2022

@author: 2512311908
"""
from dgl.data import DGLDataset
import torch
import functools
from cif_to_crystal_graph_jarvis import read_cif_jarvis
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import KFold

def get_dataloader(dataset,indices,batch_size,
                   num_workers=0, pin_memory=False):
    sampler = SubsetRandomSampler(indices)
    dataloader = GraphDataLoader(dataset, batch_size=batch_size,
                                 sampler=sampler,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)
    return dataloader


class GraphDataset(DGLDataset):
    def __init__(self,
                 cif_dir,
                 bloch_dir,
                 strus,
                 mode):
        super(GraphDataset, self).__init__(name="GNR_heterojunction_dataset")

        self.cif_data = []
        self.bloch_data = []
            
        for stru in strus:
            self.cif_data.append(cif_dir + stru + '.cif')
            self.bloch_data.append(bloch_dir + stru + str(mode) +'.npy')
        
    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        cif_file = self.cif_data[idx]
        bloch_file = self.bloch_data[idx]
        graph = read_cif_jarvis(cif_file)
        bloch = np.load(bloch_file)
        return graph, torch.Tensor(np.array(bloch)),(cif_file, graph.num_nodes())

    def __len__(self):
        return len(self.cif_data)
        
    def process(self):
        pass
    
