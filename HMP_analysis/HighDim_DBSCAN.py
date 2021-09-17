#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 18:30:55 2021

@author: abrahamr
"""

#assigns cells to a cluster and saves to a csv, generates composite rgb
from sklearn.cluster import DBSCAN
import os
import pandas as pd 
import numpy as np
import pickle
from tifffile import imsave

def make_tile_rgb_clusters(pkldir,pklname,roi): #Makes tile-level rgb
    impkl = pickle.load(open(os.path.join(pkldir,pklname),'rb'))
    rgb = np.zeros((512,512,3))
    for ind,cell in enumerate(impkl):
        color=roi.iloc[ind]['color']        
        for ind,val in enumerate(color):
            rgb[:,:,ind]=rgb[:,:,ind]+(val*cell['mask'])
    return rgb

def stitch_biopsy_rgb_clusters(b,pkldir,savedir, df): #stitches together rgb of cells colored by clusters
    imsize=512
    ov = np.ceil(0.1*imsize)
    pos=os.path.join('/nfs/kitbag/CellularImageAnalysis/Lupus/High_Dim_LuN/LuN_biopsy_images/tile_positions/',str(b+'_positions.pkl'))
    positions=pickle.load(open(pos,'rb'))  
    [numrows,numcols]=positions.shape
    new_comp=np.zeros((int(numrows*(imsize-ov)),int(numcols*(imsize-ov)),3)) #Initializes rgb
    for x in range(numrows):
        for y in range(numcols):
            roi_num=positions[x,y]
            pklname=str(b+'_allclasses_'+str(roi_num)+'.pkl')
            if os.path.exists(os.path.join(pkldir,pklname)):
                roi=df.loc[df['roi_num']==roi_num]
                tile=make_tile_rgb_clusters(pkldir,pklname,roi)
            else:
                shape=new_comp[int(x*(imsize-ov)):int((x*(imsize-ov))+imsize),int(y*(imsize-ov)):int((y*(imsize-ov))+imsize)].shape
                tile=np.zeros(shape)
            new_comp[int(x*(imsize-ov)):int((x*(imsize-ov))+imsize),int(y*(imsize-ov)):int((y*(imsize-ov))+imsize),:]=tile
    newcompname=str(b+'_cells_stitched_clusters.tif')
    print(new_comp.shape)
    new_comp=new_comp.astype(np.uint8)
    try:
        imsave(os.path.join(savedir,newcompname),new_comp,imagej=True,compress=1)  
    except:
        print('Image too large!')

def main():
    # Read in csvs, do dbscan to assign clusters, add to csv -> save 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvdir", default= "CellFeats_csvs", help = "path to directory with data files (csvs)") 
    parser.add_argument("--pkldr",default="clean_pred_pkls/", help = "path to prediction pkl files")
    parser.add_argument("--compdir",default="neighborhood_composites/", help = "directory to save stitched cell composites")
    args = parser.parse_args()
    compdir=args.compdir
    if not os.path.exists(compdir):
        os.makedirs(compdir)
    csv_dir=args.csvdir
    pkldir=args.pkldir

    csvs=os.listdir(csv_dir)
    csvs=[x for x in csvs if x.endswith('.csv')]
    for biopsy in csvs:
        print(biopsy)
        b='_'.join(biopsy.split('_')[:-1])
        df=pd.read_csv(os.path.join(csv_dir,biopsy))
        centroids=np.array(df[['global_centroid_row','global_centroid_column']])
        dbscan=DBSCAN(eps=50,min_samples=2)
        clusters=dbscan.fit_predict(centroids)
        df['clusters']=clusters
        color_dict={}
        colors=[[255,0,0],[0,255,0],[0,0,255],[128,0,0],[0,128,0],[0,0,128],[255,255,0],[255,0,255],[0,255,255],[128,128,0],[128,0,128],[0,128,128]]
        i=0
        for c in df['clusters'].unique():
            color_dict.update({c:colors[i]})
            i=i+1
            if i==(len(colors)-1):
                i=0
        color_dict.update({-1:[255,255,255]})
        df['color']=[color_dict[x] for x in df['clusters']]
        stitch_biopsy_rgb_clusters(b,pkldir,compdir, df)  
        df.to_csv(os.path.join(csv_dir,str(b+'_allcells.csv')))

if __name__=='__main__':
    main()
        
    
    
    


