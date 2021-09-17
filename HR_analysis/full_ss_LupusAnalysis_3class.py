#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:55:33 2020

@author: rebaabraham
"""

### Lupus single stain cellular analysis

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import mannwhitneyu, ks_2samp
from scipy.spatial.distance import euclidean
from datetime import datetime

## Extract Accession numbers and make new column
def Extract_acc(Case_str):
    sep = '-'
    name_str = Case_str.replace('-','_')
    name_str = name_str.replace('s','S')
    name = name_str.split('_')[0:2] 
    Acc = sep.join(name)
    return Acc


## Compare list of accessions with clinical data, extract cells with matching accessions
def compare_acc(lst1,lst2, df):
    acc = list(set(lst1)&set(lst2))
    return df.loc[(df['Accession #']).isin(acc) == True]               

## Fx for extracting data from clinical spreadsheet, adding to the feature dfs, renaming column
def ExtractfromClin(clin_df,add_df,feat,featname):
    feat_dict = dict(zip(clin_df['Accession #'],clin_df[feat]))
    def check_feat(Acc):
        return feat_dict[Acc]
    add_df[featname] = add_df['Accession #'].apply(check_feat)
    return add_df 

def MakeCellDF(df, clin_dat, prob):    
    cells = pd.read_csv(df)
    cells = cells[(cells.Area<100)&(cells.Area>3)&(cells.Prob>prob)]
    cells['Accession #']=cells['Case'].apply(Extract_acc)
    ## Filters out those that are missing accessions
    cells = cells.loc[(cells['Accession #']).str.startswith('S') == True]
    # Extracts features from biopsies with clinical data
    cells = compare_acc(list(cells['Accession #'].unique()), list(clin_dat['Accession #'].unique()),cells)
    cells = ExtractfromClin(clin_dat,cells,'ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)','ESRD')
    cells.rename(columns={'Unnamed: 0':'index'}, inplace=True)
    print(cells.shape)
    print("Number of biopsies: "+str(len(cells['Accession #'].unique())))
    return cells

## Generate cell counts and ratios per biopsy, and per ROI
    ## Compare ESRD + vs -, make histograms 
    ## look at distribution of cells per ROI in ESRD + vs - 

def CellCounts(feats, Class_list, clin_dat,sd):                                       
    Cell_type_counts = feats.groupby(['Accession #','Class_id']).count()
    Cell_type_counts = Cell_type_counts.reset_index()
    Cell_type_counts = Cell_type_counts.pivot(index='Accession #', columns='Class_id',values='Area')
    Cell_type_counts = Cell_type_counts.reset_index()                                          
    Cell_type_counts = Cell_type_counts.fillna(0)    
    Cell_type_counts['total_cells']=sum([Cell_type_counts['CD20+'],Cell_type_counts['CD11c+'],Cell_type_counts['BDCA2+'],Cell_type_counts['CD3+CD4+'],Cell_type_counts['CD3+CD4-']])                     
    for i in Class_list:
        for ii in Class_list:
            if ii == i:
                pass
            else:
                ratio = Cell_type_counts[i]/Cell_type_counts[ii]
                name = str(i+'_to_'+ii) 
                Cell_type_counts[name] = ratio        
    Cell_type_counts = Cell_type_counts.replace([np.inf,np.NaN],[0,0])  
    ExtractfromClin(clin_dat,Cell_type_counts,'ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)','ESRD')
    
    Cell_type_counts.to_csv(sd+'LupusBiopsy_CellCountFeats.csv')
    return Cell_type_counts
            
def CellCountsbyROI(feats,Class_list, clin_dat): 
    Cell_type_byROI = feats.groupby(['Case','Class_id']).count()['Area']
    Cell_type_byROI = Cell_type_byROI.reset_index()
    Cell_type_byROI = Cell_type_byROI.pivot(index='Case', columns='Class_id',values='Area')
    Cell_type_byROI = Cell_type_byROI.reset_index()                                          
    Cell_type_byROI = Cell_type_byROI.fillna(0)    
    Cell_type_byROI['total_cells']=sum([Cell_type_byROI['CD20+'],Cell_type_byROI['CD11c+'],Cell_type_byROI['BDCA2+'],Cell_type_byROI['CD3+CD4+'],Cell_type_byROI['CD3+CD4-']])
    for i in Class_list:
        for ii in Class_list:
            if ii == i:
                pass
            else:
                ratio = Cell_type_byROI[i]/Cell_type_byROI[ii]
                name = str(i+'_to_'+ii) 
                Cell_type_byROI[name] = ratio        
    Cell_type_byROI = Cell_type_byROI.replace([np.inf,np.NaN],[0,0])  
    Cell_type_byROI['Accession #']=Cell_type_byROI['Case'].apply(Extract_acc)
    ExtractfromClin(clin_dat,Cell_type_byROI,'ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)','ESRD')
    return Cell_type_byROI
        
def CellCountsHistsandDiffs(df, sd, perROI,savestr):
    feats=df.columns
    feats = feats.drop(['Accession #','ESRD'])
    if 'Case' in feats:
        feats=feats.drop('Case')
    ESRD0=df.loc[df['ESRD']==0]
    ESRD1=df.loc[df['ESRD']==1]
    ESRD2=df.loc[df['ESRD']==2]
    textargs={'size':'12'}
    for feat in feats:
        Subset_list=[ESRD0[feat],ESRD1[feat],ESRD2[feat]]
        weights0=np.ones_like(ESRD0[feat])/len(ESRD0[feat])
        weights1=np.ones_like(ESRD1[feat])/len(ESRD1[feat])
        weights2=np.ones_like(ESRD2[feat])/len(ESRD2[feat])
        weights=[weights0,weights1,weights2]
        plt.clf()
        plt.hist(Subset_list, color=['blue','#fc9003','green'], bins=50, weights=weights)
        if perROI==True:
            plt.xlabel(feat,**textargs)
            plt.ylabel('Proportion of ROIs',**textargs)
            plt.title('Distribution of '+feat,**textargs)
            plt.legend(['ESRD-','ESRD+','ESRD current'])
            plt.savefig(sd+'/Histogram_'+feat+'_byESRD_perROI'+savestr+'.png')  
        elif perROI==False:
            plt.xlabel(feat,**textargs)
            plt.ylabel('Proportion of Biopsies',**textargs)
            plt.title('Distribution of '+feat,**textargs)
            plt.legend(['ESRD-','ESRD+','ESRD current'])
            plt.savefig(sd+'/Histogram_'+feat+'_byESRD'+savestr+'.png') 
        plt.show()
        u1,p1 = mannwhitneyu(ESRD0[feat],ESRD1[feat])
        d1,p2 = ks_2samp(ESRD0[feat],ESRD1[feat])
        u2,p3 = mannwhitneyu(ESRD0[feat],ESRD2[feat])
        d2,p4 = ks_2samp(ESRD0[feat],ESRD2[feat])
        u3,p5 = mannwhitneyu(ESRD2[feat],ESRD1[feat])
        d3,p6 = ks_2samp(ESRD2[feat],ESRD1[feat])
        print('Avg '+feat+'in ESRD-: '+str(ESRD0[feat].mean()))
        print('Avg '+feat+'in ESRD+: '+str(ESRD1[feat].mean()))
        print('Avg '+feat+'in current ESRD: '+str(ESRD2[feat].mean()))
        print('p value for diff between ESRD+ and ESRD- in '+feat+' by M-W: '+str(p1)+' \n')
        print('p value for diff in distributions of ESRD+ and ESRD- in '+feat+' by K-S: '+str(p2)+' \n')
        print('p value for diff between ESRD current and ESRD- in '+feat+' by M-W: '+str(p3)+' \n')
        print('p value for diff in distributions of ESRD current and ESRD- in '+feat+' by K-S: '+str(p4)+' \n')
        print('p value for diff between ESRD+ and ESRD current in '+feat+' by M-W: '+str(p5)+' \n')
        print('p value for diff in distributions of ESRD+ and ESRD current in '+feat+' by K-S: '+str(p6)+' \n')
        
def CheckFeatBoxPlot(df, sd, savestr,feat,featstr):
    ESRD0=df.loc[df['ESRD']==0]
    ESRD1=df.loc[df['ESRD']==1]
    ESRD2=df.loc[df['ESRD']==2]
    textargs={'size':'16','fontname':'Times New Roman'}
    medianlineprops={'linewidth':'0'}
    boxprops={'linewidth':'2'}
    whiskerprops={'linewidth':'2'}
    capprops={'linewidth':'2'}
    meanprops=dict(markerfacecolor='white',markeredgecolor='black',marker='D',markersize=5)
    u1,p1=mannwhitneyu(ESRD0[feat], ESRD1[feat])
    u2,p2=mannwhitneyu(ESRD0[feat], ESRD2[feat])
    u3,p3=mannwhitneyu(ESRD2[feat], ESRD1[feat])
    print(featstr)
    print(str(df.groupby('ESRD').mean()[feat]))
    print('p value diff between ESRD + and -: '+str(p1)) 
    print('p value diff between ESRD current and -: '+str(p2)) 
    print('p value diff between ESRD + and current: '+str(p3)) 
    fig = plt.figure(figsize=(4,5), frameon=False)
    ax=fig.add_axes([0.2,0.1,0.75,0.85])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    bplot=ax.boxplot([ESRD0[feat],ESRD1[feat],ESRD2[feat]],
                     labels=['ESRD-','ESRD+','ESRD current'], patch_artist=True, 
                     showmeans=True, widths=0.4,boxprops=boxprops,
                     whiskerprops=whiskerprops,medianprops=medianlineprops,
                     capprops=capprops,meanprops=meanprops)
    for patch, color in zip(bplot['boxes'],['blue','#fc9003','green']):
        patch.set_facecolor(color)
    plt.ylabel(featstr,**textargs)
    plt.xticks(fontsize=14,fontname='Times New Roman')
    plt.yticks(fontsize=14,fontname='Times New Roman')
    plt.savefig(sd+'/'+savestr+'.png')
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--df", default= "CellData_forpublication.csv", help = "path to data file (csv)") 
    parser.add_argument("--clin_dat", default="ClinicalData_forpublication.csv", help= "clinical data file")
    parser.add_argument("--svdr",default="GitTest/")
    parser.add_argument("--outfile",default="full_analysis_3class.txt")
    args = parser.parse_args()
    sd = args.svdr
    if os.path.exists(sd):
        pass
    else:
        os.makedirs(sd)
    # generates log of output 
    stdoutOrigin=sys.stdout
    sys.stdout = open(str(sd+args.outfile),'w')
    print('Analysis generated on: ')
    print('Data file: '+args.df)
    print(datetime.now())
    print('plots saved in: '+sd)
    print('Prob cutoff: 0.3')
    # Import clinical data 
    clin_dat = pd.read_csv(args.clin_dat)
    clin_dat['Follow up: years'] = clin_dat['Duration of follow up (days)']/365
    clin_dat=clin_dat.loc[clin_dat['Follow up: years']>2]
    clin_dat.loc[clin_dat['Time to ESRD (days)']<14,'ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)']=2 #classifies people imminently in ESRD in a separate group
    print('Restricted to 2 year follow up')
    # import cell features, clean data
    all_cells = MakeCellDF(args.df, clin_dat, 0.3)
    Class_list = ['CD3+CD4+','CD3+CD4-','CD20+','BDCA2+','CD11c+']
    # How many ROIs per ESRD+ vs -  
    ROI_list=pd.DataFrame(all_cells.Case.unique(),columns=['Case'])
    ROI_list['Accession #']=ROI_list['Case'].apply(Extract_acc)
    ROI_list= ExtractfromClin(clin_dat,ROI_list,'ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)','ESRD')
    print('# ESRD + (current=2), + (future=1) and - (0) ROIs')
    print(ROI_list.groupby('ESRD').count())
    # Comparison of ESRD + and - groups
    clin_dat = compare_acc(list(all_cells['Accession #'].unique()), list(clin_dat['Accession #'].unique()),clin_dat)
    ESRD_groups=clin_dat.groupby(['ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)']).count()['Follow up: years']
    ESRD_TI_groups=clin_dat.groupby(['ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)','TI score (0 - 3)']).count()['Follow up: years']
    print('Breakdown of ESRD status:')
    print(ESRD_groups)
    print('By TI')
    print(ESRD_TI_groups)
    # Makes histograms of cell counts/ratios
    print('___________')
    print('~Analysis of cell counts by biopsy~')
    if os.path.exists(os.path.join(sd,'counts_by_biopsy'))==False:
        os.mkdir(os.path.join(sd,'counts_by_biopsy'))
    Cell_type_counts = CellCounts(all_cells, Class_list, clin_dat,sd)
    CellCountsHistsandDiffs(Cell_type_counts, os.path.join(sd,'counts_by_biopsy'), False,'_all')

    print('~Analysis of cell counts by ROI~')
    if os.path.exists(os.path.join(sd,'counts_by_ROI_3class'))==False:
        os.mkdir(os.path.join(sd,'counts_by_ROI_3class'))
    else:
        pass
    Cell_type_countsbyROI = CellCountsbyROI(all_cells,Class_list, clin_dat)
    CellCountsHistsandDiffs(Cell_type_countsbyROI, os.path.join(sd,'counts_by_ROI_3class'), True,'_all')
    for c in Class_list:
        savestr=str(c+'_perROI_byOutcome')
        featstr=str(c+' per ROI')
        CheckFeatBoxPlot(Cell_type_countsbyROI, os.path.join(sd,'counts_by_ROI_3class'), savestr,c,featstr)
    CheckFeatBoxPlot(Cell_type_countsbyROI, os.path.join(sd,'counts_by_ROI_3class'), 'Totalcells_perROI_byOutcome','total_cells','Total cells per ROI') 

    sys.stdout.close()
    sys.stdout=stdoutOrigin 
    
if __name__=='__main__':
    main()
    
