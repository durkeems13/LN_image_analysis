#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:27:39 2020

@author: rebaabraham
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from datetime import datetime
from full_ss_LupusAnalysis_2class import *


def Clin_Feat_box(df, sd, feat, ylab):
    ESRD0=df.loc[df['ESRD']==0][feat].dropna()
    ESRD1=df.loc[df['ESRD']==1][feat].dropna()
    print('Avg '+feat+'-')
    print(str(round(df.groupby('ESRD').mean()[feat],3)))
    u,p=mannwhitneyu(ESRD0, ESRD1)
    print('p value: '+str(round(p,3))) 
    textargs={'size':'16','fontname':'Times New Roman'}
    medianlineprops={'linewidth':'0'}
    boxprops={'linewidth':'2'}
    whiskerprops={'linewidth':'2'}
    capprops={'linewidth':'2'}
    meanprops=dict(markerfacecolor='white',markeredgecolor='black',marker='D',markersize=5)
    plt.clf()
    fig = plt.figure(figsize=(3,5), frameon=False)
    ax=fig.add_axes([0.25,0.1,0.75,0.85])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    bplot=ax.boxplot([ESRD0,ESRD1],labels=['ESRD-','ESRD+'], 
                      patch_artist=True, showmeans=True, widths=0.4,boxprops=boxprops,
                      whiskerprops=whiskerprops,medianprops=medianlineprops,
                      capprops=capprops,meanprops=meanprops)
    for patch, color in zip(bplot['boxes'],['blue','red']):
        patch.set_facecolor(color)
    plt.ylabel(ylab,**textargs)
    plt.xticks(fontsize=14,fontname='Times New Roman')
    plt.yticks(fontsize=14,fontname='Times New Roman')
    plt.savefig(sd+feat+'.png')
    plt.show()

def Clin_Feat_hist(df, sd, feat,b,xlabel):
    ESRD0=df.loc[df['ESRD']==0]
    ESRD1=df.loc[df['ESRD']==1]
    weights0=np.ones_like(ESRD0[feat])/len(ESRD0[feat])
    weights1=np.ones_like(ESRD1[feat])/len(ESRD1[feat])
    weights=[weights0,weights1]
    print('Avg '+feat+'-')
    print(str(round(df.groupby('ESRD').mean()[feat],3)))
    u,p=mannwhitneyu(ESRD0[feat], ESRD1[feat])
    print('p value: '+str(round(p,3))) 
    textargs={'size':'16','fontname':'Times New Roman'}
    plt.clf()
    fig=plt.figure(figsize=(5.5,5),frameon=False)
    ax=fig.add_axes([0.15,0.15,0.75,0.75])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.hist([ESRD0[feat],ESRD1[feat]],color=['blue','red'], bins=b, weights=weights)
    if feat!='TI':
        ax.hist([ESRD0[feat],ESRD1[feat]],color=['blue','red'], bins=b, weights=weights)
        plt.xticks(fontsize=14,fontname='Times New Roman')
       
    else:
        plt.xticks([0,1,2,3],fontsize=14)
    plt.legend(['ESRD-','ESRD+'],framealpha=0,prop={'family':'Times New Roman','size':'16'})
    plt.xlabel(xlabel,**textargs)
    plt.ylabel('% of cases',**textargs)
    plt.yticks(fontsize=14,fontname='Times New Roman')
    plt.savefig(sd+feat+'.png')
    plt.show()




def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--df", default= "CellData_forpublication.csv", help = "path to data file (csv)") 
    parser.add_argument("--clin_dat", default="ClinicalData_forpublication.csv", help= "clinical data file")
    parser.add_argument("--svdr",default="HR_analysis/")
    parser.add_argument("--outfile",default="Clinicalmetrics.txt")
    args = parser.parse_args()
    sd = args.svdr
    if os.path.exists(sd):
        pass
    else:
        os.makedirs(sd)
    ## generates log of output 
    stdoutOrigin=sys.stdout
    sys.stdout = open(str(sd+args.outfile),'w')
    print('Analysis generated on: ')
    print('Data file: '+args.df)
    print(datetime.now())
    print('plots saved in: '+sd)
    clin_df=pd.read_csv(args.clin_dat)
    clin_df['Years of Follow-up'] = clin_df['Duration of follow up (days)']/365
    clin_df=clin_df.loc[clin_df['Years of Follow-up']>2]
    #clin_df.loc[clin_df['Time to ESRD (days)']<14,'ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)']=2 #classifies people imminently in ESRD in a separate group
    # Extracts features from biopsies with clinical data
    cells = MakeCellDF(args.df, clin_df, 0.3)
    clin_df = compare_acc(list(cells['Accession #'].unique()), list(clin_df['Accession #'].unique()),clin_df)
    ROI_list=pd.DataFrame(cells.Case.unique(),columns=['Case'])
    ROI_list['Accession #']=ROI_list['Case'].apply(Extract_acc)
    ROI_list= ExtractfromClin(clin_df,ROI_list,'ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)','ESRD')
    ROI_counts=list(ROI_list.groupby('Accession #').count()['Case'])

    Class_list = ['CD3+CD4+','CD3+CD4-','CD20+','BDCA2+','CD11c+']
    Cell_type_counts = CellCounts(cells, Class_list, clin_df,sd)
    Cell_type_counts['num_ROIs']=ROI_counts
    Cell_type_counts['total_cells']=sum([Cell_type_counts['CD20+'],Cell_type_counts['CD11c+'],Cell_type_counts['BDCA2+'],Cell_type_counts['CD3+CD4+'],Cell_type_counts['CD3+CD4-']])
    Class_list.append('total_cells')
    for c in Class_list:
        Cell_type_counts[c+'_per_ROI']=Cell_type_counts[c]/Cell_type_counts['num_ROIs']
    Cell_type_countsbyROI = CellCountsbyROI(cells,Class_list, clin_df)
    clin_df.rename(columns={'ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)':'ESRD', 'TI score (0 - 3)':'TI'}, inplace=True)                                 
    ESRD_groups=clin_df.groupby(['ESRD']).count()['Follow up: years']
    ESRD_TI_groups=clin_df.groupby(['ESRD','TI']).count()['Follow up: years']
    print('Breakdown of ESRD status:')
    print(ESRD_groups)
    print('By TI')
    print(ESRD_TI_groups)

    Clin_Feat_box(clin_df, sd,'Years of Follow-up','Follow-up Time (years)')   
    Clin_Feat_box(clin_df, sd,'Patient Age at the time of biopsy','Patient Age (years)')  
    Clin_Feat_box(clin_df, sd,'Duration of disease at the time of biopsy','Disease Duration (years)') 
    Clin_Feat_hist(clin_df, sd,'TI',4,'TI Score') 


    sys.stdout.close()
    sys.stdout=stdoutOrigin 
if __name__=='__main__':
    main()