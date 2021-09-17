#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:07:20 2021

@author: abrahamr
"""
# Generate csv of cells for each biopsy
import os 
import pandas as pd
import numpy as np
import pickle 

#Translates secondary marker expression into additional cellular subsets -> 'Subset_id'
def get_subsets(df):
    sub_dict={(2,0,0,1):6,(2,1,1,0):7,(2,1,0,0):7,(3,1,0,0):8} #maps sequence with cell class; Treg=CD4+PD1-ICOS-FoxP3+; Tfh=CD4+PD1+ICOS+/-FoxP3-;exhausted CD8=CD8+PD1+ICOS-FoxP3-
    subset_ids=[]
    for ind,cell in df.iterrows():
        check=(cell['class_id'],cell['pd1_expression'],cell['icos_expression'],cell['foxp3_expression'])
        if check in sub_dict.keys():
            subset_ids.append(sub_dict[check])
        else:
             subset_ids.append(cell['class_id'])
    df['Subset_id']=subset_ids
    return df

def generate_csv(b,pkl_list, read_dir,write_dir,file_end,pospath):
    if len(pkl_list):
        cells=[]
        cell_id=0
        pos=os.path.join(pospath,str(b+'_positions.pkl'))
        positions=pickle.load(open(pos,'rb')) 
        for pkl in pkl_list:
            with open(os.path.join(read_dir,pkl),'rb') as f:
                x=pickle.load(f)
            for cell in x:
                cols=list(cell.keys())
                vals=[]
                for k in cols:
                    vals.append(cell[k])
                #Shifting centroid to the "global" coord system
                roi_num=int(pkl.split('_')[-1].split('.')[0])
                [r,c]=np.where(positions==roi_num)
                shift_r=float(vals[3]+(r*460)) #460 represents a tile size of 512 - overlap of 0.1 
                shift_c=float(vals[4]+(c*460))
                for v in [shift_r,shift_c,roi_num, cell_id]:
                    vals.append(v)
                for i in ['global_centroid_row','global_centroid_column','roi_num','cell_id']:
                    cols.append(i)
                cell_df=pd.DataFrame([vals],columns=cols)
                cells.append(cell_df)
                cell_id=cell_id+1
        
        all_cells=pd.concat(cells,axis=0)
        all_cells=get_subsets(all_cells)
        all_cells.to_csv(os.path.join(write_dir,b+file_end))
            
            
            
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkls_read",
        type=str,
        default='clean_pred_pkls/',  
        help="path to directory with prediction pkl fils"
    )
    
    parser.add_argument(
        "--write",
        type=str,
        default='CellFeats_csvs',  
        help="path to directory to save cell csvs"
    )
    parser.add_argument(
        "--file_append",
        type=str,
        default='_allcells.csv',  
        help="ending of file name"
    )
    parser.add_argument(
        "--positions",
        type=str,
        default='tile_positions/',  
        help="path to directory to tile positions"
    )
    args= parser.parse_args() 
    read_dir=args.pkls_read
    write_dir=args.write
    pkls=os.listdir(read_dir)
    file_end=args.file_append
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    positions=os.listdir(args.positions) #directory with pickles that map the roinum to the location in the composite 
    biopsy_list=[('_').join(x.split('_')[:-1]) for x in positions] #lists each sample, with multiple samples per biopsy sometimes
    for b in biopsy_list:
        print(b)
        pkl_list=[x for x in pkls if ('_').join(x.split('_')[:-2])==b]
        pkl_list.sort()
        generate_csv(b,pkl_list,read_dir,write_dir,file_end, positions)  

if __name__=='__main__':
    main()
