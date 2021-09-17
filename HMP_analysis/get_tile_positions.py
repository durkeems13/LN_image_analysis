#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 15:44:07 2021

@author: abrahamr
"""
#Script loops through biopsies and figures out the positions of each tile, saves as a pickle file
import os
import numpy as np
from tifffile import imread
import pickle

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--svdr",default="tile_positions/", help = "path to save directory")
    parser.add_argument("--compdir",default="Semi-Auto_Composites", help = "directory with raw image composites")
    args = parser.parse_args()
    savedir=args.svdr
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    compdir=args.compdir

    biopsies=os.listdir(compdir)
    imsize=512
    ov = np.ceil(0.1*imsize)
    for b in biopsies:
        if not os.path.exists(os.path.join(savedir,b)):
            os.makedirs(os.path.join(savedir,b))
        imname=str(b+'_D1_405.tif')
        comp=imread(os.path.join(compdir,b,'D1',imname))
        [r,c]=comp.shape
        numrows = r/(imsize-ov) #this number is 512-52 (10% overlap)
        numcols = c/(imsize-ov)
        positions=np.empty((int(np.ceil(numrows)),int(np.ceil(numcols))),dtype='int')
        roi_num=0
        for x in range(int(np.ceil(numrows))):
            for y in range(int(np.ceil(numcols))):
                positions[x,y]=roi_num
                roi_num=roi_num+1
        pklname=b+'_positions.pkl'
        pickle.dump(positions,open(os.path.join(savedir,pklname),'wb')) 

if __name__=='__main__':
    main()