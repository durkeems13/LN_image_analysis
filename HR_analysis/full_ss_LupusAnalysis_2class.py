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
from scipy.stats import mannwhitneyu, ks_2samp, chi2_contingency
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
    textargs={'size':'12'}
    for feat in feats:
        Subset_list=[ESRD0[feat],ESRD1[feat]]
        weights0=np.ones_like(ESRD0[feat])/len(ESRD0[feat])
        weights1=np.ones_like(ESRD1[feat])/len(ESRD1[feat])
        weights=[weights0,weights1]
        plt.clf()
        plt.hist(Subset_list, color=['blue','red'], bins=50, weights=weights)
        if perROI==True:
            plt.xlabel(feat,**textargs)
            plt.ylabel('Proportion of ROIs',**textargs)
            plt.title('Distribution of '+feat,**textargs)
            plt.legend(['ESRD-','ESRD+'])
            plt.savefig(sd+'/Histogram_'+feat+'_byESRD_perROI'+savestr+'.png')  
        elif perROI==False:
            plt.xlabel(feat,**textargs)
            plt.ylabel('Proportion of Biopsies',**textargs)
            plt.title('Distribution of '+feat,**textargs)
            plt.legend(['ESRD-','ESRD+'])
            plt.savefig(sd+'/Histogram_'+feat+'_byESRD'+savestr+'.png') 
        plt.show()
        u,p1 = mannwhitneyu(ESRD0[feat],ESRD1[feat])
        d,p2 = ks_2samp(ESRD0[feat],ESRD1[feat])
        print('Avg '+feat+'in ESRD-: '+str(ESRD0[feat].mean()))
        print('Avg '+feat+'in ESRD+: '+str(ESRD1[feat].mean()))
        print('p value for diff in '+feat+' by M-W: '+str(p1)+' \n')
        print('p value for diff in distributions of '+feat+' by K-S: '+str(p2)+' \n')
        

## Pools observations by ESRD status and compares via Mann-Whitney, generates box plot
def CheckFeatBoxPlot(df, sd, savestr,feat,featstr):
    ESRD0=df.loc[df['ESRD']==0]
    ESRD1=df.loc[df['ESRD']==1]
    textargs={'size':'12'}
    u,p=mannwhitneyu(ESRD0[feat], ESRD1[feat])
    print(featstr)
    print(str(df.groupby('ESRD').mean()[feat]))
    print('p value: '+str(p)) 
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
    bplot=ax.boxplot([ESRD0[feat],ESRD1[feat]],labels=['ESRD-','ESRD+'], 
                      patch_artist=True, showmeans=True, widths=0.4,boxprops=boxprops,
                      whiskerprops=whiskerprops,medianprops=medianlineprops,
                      capprops=capprops,meanprops=meanprops)
    for patch, color in zip(bplot['boxes'],['blue','red']):
        patch.set_facecolor(color)
    plt.ylabel(featstr,**textargs)
    plt.xticks(fontsize=14,fontname='Times New Roman')
    plt.yticks(fontsize=14,fontname='Times New Roman')
    plt.savefig(sd+savestr+'.png')
    plt.show()

## Check cellular nearest neighbor
def CheckMinDistID(df,type1,sd, clin_dat):
    nearest_neighbors = []   
    for name, group in df.groupby('Case'):
        for index,row in group.iterrows():
            if row['Class_id'] == type1:
                real_ind_1=row['index']
                d1=[row['Centroid_x'],row['Centroid_y']]
                dists = {}
                for index2,row2 in group.iterrows():
                    if index2 != index:
                        real_ind_2=row2['index']
                        d2=[row2['Centroid_x'],row2['Centroid_y']]
                        dist=euclidean(d1,d2)
                        dists.update({dist:real_ind_2})
                if len(dists)>0:
                    min_dist = min(dists.keys())
                    ind = dists[min_dist]
                    neighbor_id = group.loc[group['index']==ind]['Class_id'].values[0]
                    newr = pd.DataFrame([[name,index,neighbor_id]],columns=['Case','index','nearest_neighbor']) 
                    nearest_neighbors.append(newr)
    nearest_neighbors_df = pd.concat(nearest_neighbors, axis=0)
    nearest_neighbors_df['Accession #']=nearest_neighbors_df['Case'].apply(Extract_acc)
    nearest_neighbors_df = ExtractfromClin(clin_dat,nearest_neighbors_df,'ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)','ESRD')
    nn_1 = nearest_neighbors_df.groupby(['ESRD','nearest_neighbor']).count()['index'][1]
    nn_0 = nearest_neighbors_df.groupby(['ESRD','nearest_neighbor']).count()['index'][0]
    nn_1= nn_1.reset_index()
    nn_0= nn_0.reset_index()
    nn_1.rename(columns={'index':'counts'}, inplace=True)
    nn_0.rename(columns={'index':'counts'}, inplace=True)
    nn_1['proportion']= nn_1['counts']/sum(nn_1['counts'])
    nn_0['proportion']= nn_0['counts']/sum(nn_0['counts'])
    nn_1['SE']= 1.96*np.sqrt((nn_1['proportion']*(1-nn_1['proportion']))/sum(nn_1['counts']))
    nn_0['SE']= 1.96*np.sqrt((nn_0['proportion']*(1-nn_0['proportion']))/sum(nn_0['counts']))
    return nn_0,nn_1

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--df", default= "CellData_forpublication.csv", help = "path to data file (csv)") 
    parser.add_argument("--clin_dat", default="ClinicalData_forpublication.csv", help= "clinical data file")
    parser.add_argument("--svdr",default="HR_analysis/")
    parser.add_argument("--outfile",default="full_analysis.txt")
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
    print('Restricted to 2 year follow up')
    # import cell features, clean data
    all_cells = MakeCellDF(args.df, clin_dat, 0.3)
    Class_list = ['CD3+CD4+','CD3+CD4-','CD20+','BDCA2+','CD11c+']
    APC_list = ['CD20+','BDCA2+','CD11c+']
    T_list = ['CD3+CD4+','CD3+CD4-']

    # How many ROIs per ESRD+ vs -  
    ROI_list=pd.DataFrame(all_cells.Case.unique(),columns=['Case'])
    ROI_list['Accession #']=ROI_list['Case'].apply(Extract_acc)
    ROI_list= ExtractfromClin(clin_dat,ROI_list,'ESRD on dialysis, transplant or cr >6 (no = 0, yes = 1)','ESRD')
    print('# ESRD + vs - ROIs')
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
    CellCountsHistsandDiffs(Cell_type_counts, os.path.join(sd,'counts_by_biopsy/'), False,'_all')
    CheckFeatBoxPlot(Cell_type_counts, os.path.join(sd,'counts_by_biopsy/'), 'Totalcells_perbiopsy_byOutcome','total_cells','Total cells per sample') 
    
    
    print('~Analysis of cell counts by ROI~')
    if os.path.exists(os.path.join(sd,'counts_by_ROI'))==False:
        os.mkdir(os.path.join(sd,'counts_by_ROI'))
    else:
        pass
    Cell_type_countsbyROI = CellCountsbyROI(all_cells,Class_list, clin_dat)
    print(Cell_type_countsbyROI.groupby('ESRD').count()['total_cells'])
    all_cells=all_cells.loc[all_cells['Case'].isin(Cell_type_countsbyROI['Case'])]
    CellCountsHistsandDiffs(Cell_type_countsbyROI, os.path.join(sd,'counts_by_ROI'), True,'_all')

    for c in Class_list:
        savestr=str(c+'_perROI_byOutcome.png')
        featstr=str(c+' per ROI')
        CheckFeatBoxPlot(Cell_type_countsbyROI, os.path.join(sd,'counts_by_ROI/'), savestr,c,featstr)
    CheckFeatBoxPlot(Cell_type_countsbyROI, os.path.join(sd,'counts_by_ROI/'), 'Totalcells_perROI_byOutcome','total_cells','Total cells per ROI') 
    
    # Nearest Neighbor Analysis
    print('___________')
    print('~Nearest neighbor analysis~')
 
    if os.path.exists(os.path.join(sd,'NNAnalysis_all')) != True:
        os.mkdir(os.path.join(sd,'NNAnalysis_all'))
    Class_list = ['CD3+CD4+','CD3+CD4-','CD20+','BDCA2+','CD11c+']
    
    nns={}
    for c in Class_list:
        nn_0,nn_1=CheckMinDistID(all_cells,c,os.path.join(sd,'NNAnalysis_all'), clin_dat)
        nns.update({c:[nn_0,nn_1]})
    for b in Class_list:
        vals0=[]
        vals1=[]
        chi2_pvals=[]
        for c in Class_list:
            nn_0=nns[c][0]
            nn_1=nns[c][1]
            nn_c0=sum(nn_0.loc[nn_0['nearest_neighbor']==b]['counts'])
            nn_c1=sum(nn_1.loc[nn_1['nearest_neighbor']==b]['counts'])
            total_c0=sum(nn_0['counts'])
            total_c1=sum(nn_1['counts'])
            vals0.append(nn_0.loc[nn_0['nearest_neighbor']==b])
            vals1.append(nn_1.loc[nn_1['nearest_neighbor']==b])
            #makes contingency table with counts
            cont=np.array([[nn_c0,total_c0-nn_c0],[nn_c1,total_c1-nn_c1]])
            chi2,p,dof,expected=chi2_contingency(cont)
            chi2_pvals.append(p)
        vals0=pd.concat(vals0)
        vals1=pd.concat(vals1)
        vals0['Class']=Class_list
        vals1['Class']=Class_list  
        X = vals0['Class']
        width = 0.35
        textargs={'size':'14','fontname':'Times New Roman'}
        plt.clf()
        fig=plt.figure(figsize=(6,5),frameon=False)
        ax=fig.add_axes([0.15,0.15,0.75,0.75])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.bar(X,vals0['proportion'], -width, align='edge',color='blue',yerr=vals0['SE'], capsize=3)
        ax.bar(X,vals1['proportion'], +width, align='edge',color='red',yerr=vals1['SE'], capsize=3)
        plt.xticks(fontsize=12,fontname='Times New Roman')
        plt.yticks(fontsize=12,fontname='Times New Roman')
        plt.legend(['ESRD -','ESRD +'],framealpha=0,prop={'family':'Times New Roman','size':'16'})
        plt.ylabel('Frequency of ' + b+ ' as Nearest neighbors',**textargs)
        plt.savefig(os.path.join(sd,'NNAnalysis_all',b+'_Nearest_neighbors.png'))
        chi=pd.DataFrame(zip(Class_list,chi2_pvals),columns=['Class','chi2_p'])
        print('nearest neighbors in ESRD-')
        print(vals0)
        print('nearest neighbors in ESRD+')
        print(vals1)
        print('Chi-square p values')
        print(chi)
     
    sys.stdout.close()
    sys.stdout=stdoutOrigin 
    
if __name__=='__main__':
    main()
    
