# -*- coding: utf-8 -*-
"""
Created on Mon May 23 22:37:07 2022

@author: 2512311908
"""
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
import numpy as np
from dgl import save_graphs, load_graphs

def read_cif_jarvis(cif_file):
    structure = Atoms.from_cif(cif_file, use_cif2cell=False)
    graph = Graph.atom_dgl_multigraph(structure, compute_line_graph = False)
    graph.ndata["frac_coords"] = torch.Tensor(np.array(structure.coords))
    graph.ndata['atomic_numbers'] = torch.IntTensor(np.array(structure.atomic_numbers))
    
    #graph_path = cif_file+ '_dgl_graph_pure.bin'
    #save_graphs(graph_path, graph)
 
    return graph    



