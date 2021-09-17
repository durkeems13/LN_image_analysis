#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:51:01 2021

@author: abrahamr
"""
# What proportion of cells are in aggregates vs singlets
# histogram of aggregate sizes within high chronicity vs low chronicity patients 

import pandas as pd
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle 
import sys
from scipy.stats import mannwhitneyu, pearsonr, ks_2samp

def get_percluster_counts_base(b,all_cells_df,s):
    df=all_cells_df.groupby(['clusters','class_id']).count()
    df=df.reset_index()
    df=df.pivot(index='clusters',columns='class_id',values='cell_id')
    df=df.replace(np.NaN, 0)
    for x in range(1,6):
        if x not in df.columns:
            df[x]=np.repeat(0,df.shape[0])
    df=df[list(range(1,6))]      
    df=df.reset_index()
    df.columns=['clusters','DN','CD4+','CD8+','CD20+','CD138+']
    df['total_cells']=df['DN']+df['CD4+']+df['CD8+']+df['CD20+']+df['CD138+']
    df['Accession #']=b
    df['section']=s
    return df


def ExtractfromClin(clin_df,add_df,feat,featname):
    feat_dict = dict(zip(clin_df['Accession #'],clin_df[feat]))
    def check_feat(Acc):
        return feat_dict[Acc]
    add_df[featname] = add_df['Accession #'].apply(check_feat)
    return add_df 

def CheckFeatBoxPlot(df, sd, savestr,feat,featstr):
    high_c=df.loc[df['chronicity_index']>=4]
    low_c=df.loc[df['chronicity_index']<4] 
    if (high_c.shape[0]>0)&(low_c.shape[0]>0):
        textargs={'size':'12'}
        u,p=mannwhitneyu(low_c[feat], high_c[feat])
        print(featstr)
        print('High chronicity: '+str(high_c.mean()[feat]))
        print('Low chronicity: '+str(low_c.mean()[feat]))
        print('p value: '+str(p)) 
        textargs={'size':'16','fontname':'Times New Roman'}
        medianlineprops={'linewidth':'0'}
        boxprops={'linewidth':'2'}
        whiskerprops={'linewidth':'2'}
        capprops={'linewidth':'2'}
        meanprops=dict(markerfacecolor='white',markeredgecolor='black',marker='D',markersize=5)
        plt.clf()
        fig = plt.figure(figsize=(4,5), frameon=False)
        ax=fig.add_axes([0.25,0.1,0.75,0.85])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        bplot=ax.boxplot([low_c[feat],high_c[feat]],labels=['Low chronicity','High chronicity'], 
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

def plot2D(counts,t1,t2,x,savedir,featstr1,featstr2,savestr): #2D plot, no colors
    R,p=pearsonr(counts[t1],counts[t2])
    print('Correlation between '+t1+' and '+t2+'in'+savestr)
    print('R: '+str(R)+', p: '+str(p))
    R_text='Pearson r: '+str(round(R,3))
    y=min(counts[t2])+0.5
    textargs={'size':'16','fontname':'Times New Roman'}
    plt.clf()
    fig = plt.figure(figsize=(4,4), frameon=False)
    ax=fig.add_axes([0.2,0.1,0.75,0.85])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.scatter(counts[t1],counts[t2],alpha=0.7,color='black',s=20)
    plt.xlabel(featstr1,**textargs)
    plt.ylabel(featstr2,**textargs)
    plt.xticks(fontsize=14,fontname='Times New Roman')
    plt.yticks(fontsize=14,fontname='Times New Roman')
    plt.text(x,y,R_text,**textargs)
    plt.savefig(os.path.join(savedir,str(savestr+'_'+t1+'_'+t2+'.png')))
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvdir", default= "CellFeats_csvs", help = "path to directory with data files (csvs)") 
    parser.add_argument("--svdr",default="ClusterAnalysis/", help = "path to save directory")
    parser.add_argument("--outfile",default="clusteranalysis.txt", help = "text file to save outputs to")
    args = parser.parse_args()
    savedir=args.svdr
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    csv_dir=args.csvdir
   
    outfile=os.path.join(savedir,args.outfile)

    stdoutOrigin=sys.stdout
    sys.stdout = open(str(outfile),'w')
    csvs=os.listdir(csv_dir)
    csvs=[x for x in csvs if x.endswith('.csv')]
    biopsy_list=[x.split('_')[0] for x in csvs]
    biopsy_list=list(np.unique(biopsy_list))

    ##Read in all data, extract all cells, and information about the clusters
    cluster_sizes_all={}
    total_cells_all={}
    every_cell=[]
    cluster_count_list_b=[]
    for b in biopsy_list:
        samples=[x for x in csvs if x.startswith(b)==True]
        if len(samples)>1:
            total_cells=0
            cluster_sizes=[]
            for biopsy in samples:
                s='_'.join(biopsy.split('_')[:-1])
                df=pd.read_csv(os.path.join(csv_dir,biopsy))
                every_cell.append(df)
                total_cells=total_cells+df.shape[0]
                c_s=[]
                for name, group in df.groupby('clusters'):
                    size=group.shape[0]
                    if name != -1:
                        c_s.append(size)
                cluster_sizes=cluster_sizes+c_s
                cluster_counts_b=get_percluster_counts_base(b,df,s)
                cluster_count_list_b.append(cluster_counts_b)
            cluster_sizes_all.update({b:cluster_sizes})
            total_cells_all.update({b:total_cells})
        else:
            s='_'.join(samples[0].split('_')[:-1])
            df=pd.read_csv(os.path.join(csv_dir,samples[0]))
            every_cell.append(df)
            total_cells=df.shape[0]
            cluster_sizes=[]
            for name, group in df.groupby('clusters'):
                size=group.shape[0]
                if name != -1:
                    cluster_sizes.append(size)
            cluster_sizes_all.update({b:cluster_sizes})
            cluster_counts_b=get_percluster_counts_base(b,df,s)
            cluster_count_list_b.append(cluster_counts_b)
            total_cells_all.update({b:total_cells})
    cluster_count_base_all=pd.concat(cluster_count_list_b,axis=0)
    #Relationship between total cell counts and number of aggregates
    agg_counts=pd.DataFrame.from_dict(total_cells_all,orient='index')
    agg_counts=agg_counts.reset_index()
    agg_counts.columns=['Accession #','total_cells']
    agg_counts['cluster_count']=[len(cluster_sizes_all[x]) for x in agg_counts['Accession #']]

    plot2D(agg_counts,'total_cells','cluster_count',10000,savedir,'Total Cells','Cluster Count','SampleLevel')

    
     #Singlet analysis
    singlets=cluster_count_base_all.loc[cluster_count_base_all['clusters']==-1]
    Tot_sing=sum(singlets['total_cells'])
    counts=[]
    for c in ['DN', 'CD4+', 'CD8+', 'CD20+', 'CD138+']:
        tot=sum(singlets[c])
        counts.append(tot)
    sing_count=pd.DataFrame(zip(['DN', 'CD4+', 'CD8+', 'CD20+', 'CD138+'],counts),columns=['class_id','counts'])
    sing_count['percent']=(sing_count['counts']/Tot_sing)*100
    print('total cell counts and proportions of only singlets')
    print(sing_count)
    color_list=['#0bb81f','#b80b0b','#0425de', '#ba04de','#04cfde']
    textprops={'fontsize':'16','fontname':'Times New Roman','fontweight':'bold'}
    plt.clf()
    fig = plt.figure(figsize=(5,4.75), frameon=False)
    ax=fig.add_axes([0.1,0,0.75,0.75])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.pie(sing_count['percent'], labels=sing_count['class_id'],autopct=None,colors=color_list,textprops=textprops,pctdistance=0.85)              
    plt.savefig(savedir+'Singlet_ClassBreakdown_prob.png')
    plt.show()
    
    # Doublet Analysis 
    doublets=cluster_count_base_all.loc[(cluster_count_base_all['total_cells']==2)&(cluster_count_base_all['clusters']!=-1)]
    Tot_doub=sum(doublets['total_cells'])
    counts=[]
    for c in ['DN', 'CD4+', 'CD8+', 'CD20+', 'CD138+']:
        tot=sum(doublets[c])
        counts.append(tot)
    doublet_counts=pd.DataFrame(zip(['DN', 'CD4+', 'CD8+', 'CD20+', 'CD138+'],counts),columns=['class_id','counts'])
    doublet_counts['percent']=(doublet_counts['counts']/Tot_doub)*100
    print(doublet_counts)
    color_list=['#0bb81f','#b80b0b','#0425de', '#ba04de','#04cfde']
    textprops={'fontsize':'16','fontname':'Times New Roman','fontweight':'bold'}
    plt.clf()
    fig = plt.figure(figsize=(5,4.75), frameon=False)
    ax=fig.add_axes([0.1,0,0.75,0.75])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.pie(doublet_counts['percent'], labels=doublet_counts['class_id'],autopct=None,colors=color_list,textprops=textprops,pctdistance=0.85)              
    plt.savefig(savedir+'Doublet_ClassBreakdown_prob.png')
    plt.show()
    
    ## Quantify the pairwise interactions
    pair_counts=[]
    for x in ['DN', 'CD4+', 'CD8+', 'CD20+', 'CD138+']:
        pc=[]
        for y in ['DN', 'CD4+', 'CD8+', 'CD20+', 'CD138+']:
            if x != y:
                P=doublets.loc[(doublets[x]==1)&(doublets[y]==1)]
                pc.append(P.shape[0]/doublets.shape[0])
            else:
                P=doublets.loc[doublets[x]==2]
                pc.append(P.shape[0]/doublets.shape[0])
        pair_counts.append(pc)
    
    PC=pd.DataFrame(pair_counts,columns=['DN', 'CD4+', 'CD8+', 'CD20+', 'CD138+'])
    PC.insert(0,'class_id',['DN', 'CD4+', 'CD8+', 'CD20+', 'CD138+'])
    mask = np.ones((5,5))
    mask[np.tril_indices_from(mask)] = False
    
    plt.clf()
    fig = plt.figure(figsize=(5,4.75), frameon=False)
    ax=fig.add_axes([0.1,0.1,0.75,0.75])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax=sns.heatmap(pair_counts,square=True,mask=mask)
    plt.savefig(savedir+'Doublet_heatmap.png')
    plt.show()
    #Numerical Histogram of Agg size
    all_clusters=cluster_count_base_all.loc[cluster_count_base_all['clusters']!=-1]
    textargs={'fontsize':12}
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,frameon=False)
    fig.subplots_adjust(hspace=0.09)
    ax1.hist(all_clusters['total_cells'],bins=50,color=['purple'])
    ax2.hist(all_clusters['total_cells'],bins=50,color=['purple'])
    ax2.set_ylim(0,850)
    ax1.set_ylim(5000,13000)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax1.xaxis.set_visible(False)
    plt.xticks(**textargs)
    plt.yticks(**textargs)
    plt.xlabel('Size of Aggregate')
    plt.ylabel('Number of Aggregates')
    plt.savefig(savedir+'/AggSize_all.png')
    fig.show()
    
    
    
    
    sys.stdout.close()
    sys.stdout=stdoutOrigin
    
if __name__=='__main__':
    main()
        
        
        
        
        