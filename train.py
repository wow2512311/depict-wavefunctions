# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:45:41 2022

@author: 2512311908
"""
import torch
import time
import argparse
from graph_dataset import GraphDataset, get_dataloader
from model import Model_on_ALIGNN, Model_on_GCN, Model_on_SAGE, Model_on_CGCNN

from torch.optim.lr_scheduler import MultiStepLR
from numpy import mean
import pytorch_msssim
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
import random
#import wandb
#wandb.init(project=" ")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())

import os
def full_stru_name(file_dir): 
    L=[] 
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.cif':
                L.append(os.path.splitext(file)[0])
    return L

def train(train_loader, model, optimizer, epoch, distance, loss_fun, sigma):
    model.train()
    all_loss = []
    for batched_graph, batched_bloch, (cif_file, num_nodes) in tqdm(train_loader):
        t1 = time.time()
        batched_graph = batched_graph.to(device)
        batched_bloch = batched_bloch.to(device)
        distance = distance.to(device)
        
        feats = batched_graph.ndata['atom_features'] #shape:(N1+N2+N3+N..)*92
        atoms_valence = batched_graph.ndata['atomic_numbers']
        atoms_valence = (atoms_valence-2).clamp(1,10).unsqueeze(1).unsqueeze(1)
        #print(atoms_valence.shape) #shape:(N1+N2+N3+N..)*1*1
        
        logits = model(batched_graph)
        #logits = model(batched_graph,feats) #shape:(N1+N2+N3+N..)*100*100
        
        frac_coords = batched_graph.ndata['frac_coords'] #shape:(N1+N2+N3+N..)*3
        frac_coords = frac_coords[:,1:].unsqueeze(1).unsqueeze(1) #shape:(N1+N2+N3+N..)*1 *1 *2
        #distance: 1*100*100*2
        temp = torch.exp(-((frac_coords-distance) ** 2) / (2*sigma)) 
        #shape:(N1+N2+N3+N..)*100*100*2
        temp = temp / temp.max()
        temp = temp.sum(axis=3).to(device) #shape:(N1+N2+N3+N..)*100*100
        
        logits = logits * temp * atoms_valence 
        
        batch_size = num_nodes.shape[0]
        #print(batch_size)
        loss = 0
        separated_logits = torch.split(logits, list(num_nodes), dim=0)
        separated_bloch = torch.chunk(batched_bloch, batch_size, dim=0)
        
        for i in range(batch_size):
            per_atom_bloch_logits = separated_logits[i].sum(axis=0)
            per_atom_bloch_logits = per_atom_bloch_logits / per_atom_bloch_logits.max()
            per_atom_bloch = separated_bloch[i]
            per_atom_bloch = per_atom_bloch / per_atom_bloch.max()
            
            loss = loss + 1 - loss_fun(per_atom_bloch_logits.unsqueeze(0).unsqueeze(0), per_atom_bloch.unsqueeze(0))
        loss = loss / batch_size
        #loss = abs(logits - batched_bloch).sum()
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        t2 = time.time()
        
        print("#######{}#####" .format(cif_file))
        print("Epoch {:05d}  TrainSet_Loss {:.6f} |Time:{:.6f} " .format(epoch, \
                                        loss.item(), (t2-t1)))
        all_loss.append(loss.item())

    mean_loss = mean(all_loss)
    #average loss in this epoch
    print("Train loss: {:.6f}".format(mean_loss))
    return mean_loss
        
def validate(val_loader, model, epoch, distance, loss_fun, sigma):
    model.eval()
    with torch.no_grad():
        all_loss = []
        for batched_graph, batched_bloch, (cif_file, num_nodes) in tqdm(val_loader):
            t1 = time.time()
            batched_graph = batched_graph.to(device)
            batched_bloch = batched_bloch.to(device)
            distance = distance.to(device)

            feats = batched_graph.ndata['atom_features']
            atoms_valence = batched_graph.ndata['atomic_numbers']
            atoms_valence = (atoms_valence-2).clamp(1,10).unsqueeze(1).unsqueeze(1)
            
            logits = model(batched_graph)
            #logits = model(batched_graph,feats)     
            
            frac_coords = batched_graph.ndata['frac_coords']
            frac_coords = frac_coords[:,1:].unsqueeze(1).unsqueeze(1)
            temp = torch.exp(-((frac_coords-distance) ** 2) / (2*sigma))
            temp = temp / temp.max()
            temp = temp.sum(axis=3)
        
            logits = logits * temp * atoms_valence
            
            batch_size = num_nodes.shape[0]
            loss = 0
            
            separated_logits = torch.split(logits, list(num_nodes), dim=0)
            separated_bloch = torch.chunk(batched_bloch, batch_size, dim=0)
            
            for i in range(batch_size):
                per_atom_bloch_logits = separated_logits[i].sum(axis=0)
                per_atom_bloch_logits = per_atom_bloch_logits / per_atom_bloch_logits.max()
                per_atom_bloch = separated_bloch[i]
                per_atom_bloch = per_atom_bloch / per_atom_bloch.max()
                
                loss = loss + 1 - loss_fun(per_atom_bloch_logits.unsqueeze(0).unsqueeze(0), per_atom_bloch.unsqueeze(0))
            loss = loss / batch_size
            #loss = abs(logits - batched_bloch).sum()

            t2 = time.time()
            
            print("################Validate#########################")
            print("#######{}#####" .format(cif_file))            
            print("Epoch:{:05d} ValidateSet_Loss {:.6f}|Time:{:.6f}".format(\
                                                    epoch, loss.item(), (t2-t1)))
            all_loss.append(loss.item())
            
        mean_loss = mean(all_loss)
        print("Validate loss: {:.6f}".format(mean_loss))
    return mean_loss

def test(test_loader, model, distance, loss_fun, sigma, final_map_size, kfold_i):
    model.eval()
    with torch.no_grad():
        all_loss = []
        i = 0
        for batched_graph, batched_bloch, (cif_file, num_nodes) in tqdm(test_loader):
            batched_graph = batched_graph.to(device)
            batched_bloch = batched_bloch.to(device)
            distance = distance.to(device)
        
            feats = batched_graph.ndata['atom_features']
            atoms_valence = batched_graph.ndata['atomic_numbers']
            atoms_valence = (atoms_valence-2).clamp(1,10).unsqueeze(1).unsqueeze(1)
            
            logits = model(batched_graph)
            #logits = model(batched_graph,feats)  
           
            frac_coords = batched_graph.ndata['frac_coords']
            frac_coords = frac_coords[:,1:].unsqueeze(1).unsqueeze(1)
            temp = torch.exp(-((frac_coords-distance) ** 2) / (2*sigma))
            temp = temp / temp.max()
            temp = temp.sum(axis=3)
        
            logits = logits * temp * atoms_valence
            
            logits = logits.sum(axis=0)
            
            logits = logits / logits.max()
            batched_bloch = batched_bloch / batched_bloch.max()
            
            loss = 1 - loss_fun(logits.unsqueeze(0).unsqueeze(0), batched_bloch.unsqueeze(0))
            #loss = abs(logits - batched_bloch).sum()
            
            np.save(cif_file[0][:-4] + str(kfold_i) +'kfold_logits.npy', logits.reshape(final_map_size,final_map_size).cpu())
            np.save(cif_file[0][:-4] + str(kfold_i) +'kfold_bloch.npy', batched_bloch.reshape(final_map_size,final_map_size).cpu())
            """
            logits_map = np.load(cif_file[0][:-4] + str(kfold_i) +'kfold_logits.npy')
            bloch_map = np.load(cif_file[0][:-4] + str(kfold_i) +'kfold_bloch.npy')

            sns_plot = sns.heatmap(logits_map, cmap='Reds',vmin=0,xticklabels=30,yticklabels=30)
            fig = sns_plot.get_figure()
            fig.savefig(cif_file[0][:-4] + str(kfold_i) +'kfold_logits.png', dpi=1500)
            plt.clf()

            sns_plot = sns.heatmap(bloch_map, cmap='Reds',vmin=0,xticklabels=30,yticklabels=30)
            fig = sns_plot.get_figure()
            fig.savefig(cif_file[0][:-4] + str(kfold_i) +'kfold_bloch.png', dpi=1500)
            plt.clf()
            """
            i = i + 1
            all_loss.append(loss.item())
            print("#######{}#####" .format(cif_file))
            print("Test loss: {:.6f}".format(loss))
            
        mean_loss = mean(all_loss)
        print("Total Mean Test loss: {:.6f}".format(mean_loss))
    return mean_loss

parser = argparse.ArgumentParser(description='Depicting Wavefunctions')
parser.add_argument('--raw-cif-dataset', type=str,default="       /dataset/cif/")
parser.add_argument('--raw-bloch-dataset', type=str,default="          /dataset/vb/")
parser.add_argument('--save_path', type=str,default="        ")
parser.add_argument('--test_ratio', default=0.1, type=float)
parser.add_argument('--k_fold', default=5, type=int)
parser.add_argument('--sigma', default=0.5, type=float)

parser.add_argument('--final_map_size', default=10000, type=int)
parser.add_argument('--dataset_randomseed', default=100, type=int)
parser.add_argument('--mode', default='-VB', type=str, help='choose a band, -VB or -CB')
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--epochs', default=100, type =int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--lr-milestones', default=[290], type=int,\
                    help='milestones for scheduler (default:[100])')
parser.add_argument('--optim', default='Adam', type=str,
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
args = parser.parse_args()
  
def main():
    final_map_size = int(np.sqrt(args.final_map_size))
    distance = torch.rand([1,final_map_size,final_map_size,2])
    for i in range(distance.shape[1]):
        for j in range(distance.shape[2]):
            distance[0,i,j,:] = torch.Tensor([(i+1)/distance.shape[1],(j+1)/distance.shape[2]])
    distance = distance.to(device)
    
    random_seed = args.dataset_randomseed
    random.seed(random_seed)
    
    strus = full_stru_name(args.raw_cif_dataset)
    random.shuffle(strus)    
    
    dataset = GraphDataset(cif_dir=args.raw_cif_dataset,
                           bloch_dir=args.raw_bloch_dataset,
                           strus=strus,
                           mode=args.mode)
    
    total_size = len(strus)
    indices = list(range(total_size))
    test_size = int(args.test_ratio * total_size)

    kf = KFold(n_splits=args.k_fold)
    
    test_indices = np.array(indices[-test_size:])
    test_loader = get_dataloader(dataset = dataset,
                                 indices = test_indices,
                                 batch_size = 1)
    
    train_val_indices = np.array(indices[:-test_size])
    
    kfold_i = 0
    final_val_loss = []
    final_test_loss = {}
    
    for train_indices, val_indices in kf.split(train_val_indices):
        k_fold_train_loss = []
        k_fold_val_loss = []
        
        train_loader = get_dataloader(dataset=dataset,
                                      indices=train_indices,
                                      batch_size=args.batch_size)
        val_loader = get_dataloader(dataset=dataset,
                                    indices=val_indices,
                                    batch_size=1)
        
        #model = Model_on_GCN()
        model = Model_on_ALIGNN()
        #model = Model_on_SAGE()
        #model = Model_on_CGCNN()
        model = model.to(device)
        loss_func = pytorch_msssim.SSIM()
        
        if args.optim == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), args.lr)
        elif args.optim == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), args.lr)
        else:
            raise NameError('Only SGD or Adam is allowed as --optim')
        
        scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
        
        
        for epoch in range(args.epochs):
            train_loss_per_epoch = train(train_loader=train_loader, model=model, optimizer=optimizer,
                                         epoch=epoch,distance=distance,loss_fun=loss_func,sigma=args.sigma)
            val_loss_per_epoch = validate(val_loader=val_loader, model=model, epoch=epoch,
                                          distance=distance,loss_fun=loss_func,sigma=args.sigma)
            scheduler.step()
            
            k_fold_train_loss.append(train_loss_per_epoch)
            k_fold_val_loss.append(val_loss_per_epoch)
            
            #wandb.log({'train_loss':train_loss_per_epoch,'val_loss':val_loss_per_epoch})
            
            if epoch == 90:
                final_val_loss.append(val_loss_per_epoch)
                print('---------Evaluate Model on Test Set---------------')
                print("#########Accuracy in test dataset########")
                test_loss = test(test_loader=test_loader, 
                                 model=model,
                                 distance=distance, 
                                 loss_fun=loss_func,
                                 final_map_size=final_map_size,
                                 sigma=args.sigma,
                                 kfold_i=kfold_i)
                final_test_loss[str(kfold_i)] = test_loss
              
            #save model per fixed epochs
            if epoch != 0 and (epoch) % 90 == 0:
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, args.save_path+str(epoch)+'NN_trained'+str(kfold_i)+'kfold.pth')
            '''
            if epoch == 0:
                best_error = val_loss_per_epoch   
            if val_loss_per_epoch < best_error:
                best_error = min(val_loss_per_epoch, best_error)
                torch.save({
                            'epoch': epoch+1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, args.modelpath+str(kfold_i)+'_best_NN_trained.pkl')'''
        
        kfold_i = kfold_i + 1    
        
    final_average_loss = mean(final_val_loss)
    print("final val loss")
    print(final_val_loss)
    print("Val loss: {:.6f}".format(final_average_loss))
    
    print(len(train_loader),len(val_loader),len(test_loader))
    
    '''
    print('---------Evaluate Model on Test Set---------------')
    print("#########Accuracy in test dataset########")
    print(final_test_loss)
    final_average_test_loss = mean(final_test_loss)
    print("Test loss: {:.6f}".format(final_average_test_loss))'''
    
    import json
    dict_json = json.dumps(final_test_loss)
    with open(args.raw_cif_dataset + 'test_loss.json', 'w+') as file:
        file.write(dict_json)
    #content=json.loads(content)

    
if __name__ == '__main__':
    main()







