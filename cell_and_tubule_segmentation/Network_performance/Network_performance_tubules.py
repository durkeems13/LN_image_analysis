#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:29:34 2021

@author: abrahamr
"""
import numpy as np
import pandas as pd
from tifffile import imread
import os
 #Tubule semantic seg Network Perf
 
 #Reads in binary gt, binary prediction, and compares the two
 

#Calculate metrics 
def Metrics(gt,pred):
    comp=gt+pred
    TP=len(np.where(comp==6)[0])
    FP=len(np.where(comp==5)[0])
    TN=len(np.where(comp==0)[0])
    FN=len(np.where(comp==1)[0])
    if (TP+FN)>0:
        Sensitivity = TP/(TP+FN)    
    else: 
        Sensitivity = 'NaN'   
    if (FP+TN)>0:
        Specificity = TN/(FP+TN)   
        FPR = FP/(FP+TN)
    else: 
        Specificity = 'NaN'      
        FPR = 'NaN'
    if (TP+FP)>0:
        Prec = TP/(TP+FP)
    else:
        Prec = 'NaN'
    if np.count_nonzero(comp):
        IOU = TP/np.count_nonzero(comp)
    else:
        IOU = 'NaN'

    return Sensitivity, Specificity, FPR , Prec, IOU
    
def evaluate(imlist,y_dir, pred_dir):
    sens_list=[]
    spec_list=[]
    fpr_list=[]
    Prec_list=[]
    iou_list=[]
    for i in imlist:
        gt=imread(os.path.join(y_dir,i))
        gt=gt/255
        pred=imread(os.path.join(pred_dir,i))
        pred=(pred[:,:,0]/255)*5
        Sensitivity, Specificity, FPR , Prec, IOU = Metrics(gt,pred)
        if Sensitivity != 'NaN':
            sens_list.append(Sensitivity)
        if Specificity != 'NaN':
            spec_list.append(Specificity)
        if FPR != 'NaN':
            fpr_list.append(FPR)
        if Prec != 'NaN':
            Prec_list.append(Prec)
        if IOU != 'NaN':
            iou_list.append(IOU)
    avg_sens=np.mean(sens_list)
    avg_spec=np.mean(spec_list)
    avg_fpr=np.mean(fpr_list)
    avg_prec=np.mean(Prec_list)
    avg_iou=np.mean(iou_list)
    metrics=pd.DataFrame([[avg_sens,avg_spec,avg_fpr,avg_prec,avg_iou]],columns=['avg_sens','avg_spec','avg_fpr','avg_prec','avg_iou'])
    return metrics    

def main():
    import argparse 
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--gt",
        type=str,
        default='/nfs/kitbag/CellularImageAnalysis/Lupus/High_Dim_LuN/cell_segmentation/data/ground_truth/tubule_gt_binary/',
        help=""
    )
    parser.add_argument(
        "--preds",
        type=str,
        default='../predictions_tubules/MP_tubules_stride16/unlabeled_testset/processing/final_overlays_rgb/',
        help=""
    )
    parser.add_argument(
        "--csvname",
        type=str,
        default='tubule_network_perf_stride16.csv',
        help=""
    )

    args, unparsed = parser.parse_known_args()
    
    y_dir=args.gt
    pred_dir=args.preds
    csvname=args.csvname
    imlist=os.listdir(pred_dir)
    
    metrics=evaluate(imlist,y_dir, pred_dir)
    metrics.to_csv(csvname)

if __name__=='__main__':
    main()

 
