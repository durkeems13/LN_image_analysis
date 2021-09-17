#import analysis as an
#import tensorflow as tf
import importlib as imp
import numpy as np
from matplotlib import pyplot as plt
import os,csv,sys,pprint,time,operator
import pickle as pkl
import itertools
# use imp.reload(an) to reload analysis
from joblib import Parallel, delayed
import pandas as pd
import multiprocessing
import argparse

from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--trainlogdir",
        type=str,
        default='',
        help=""
    )

    args, unparsed = parser.parse_known_args()

    #for network inferences
    read_folder=os.path.join('../../',args.trainlogdir,'processing/output_feats')

    cases=os.listdir(read_folder)
    auto_props=[]
    for i in range(len(cases)):
        case=cases[i]
        #print(case+'___'+str(i))
        props=pkl.load(open(os.path.join(read_folder,case),'rb'))
        auto_props.append(props)
    print('props',auto_props)
    #auto_props=list(itertools.chain(*auto_props))
    #feature_list_auto=['Area','Case','Centroid_x','Centroid_y','Circularity','Class_id','Convex_area','Convex_perimeter','Eccentricity','Equivalent_diameter','Major_axis_length', 'Major_minor_axis_ratio','CD20+_mean_min_dist','CD3+CD4+_mean_min_dist','CD3+CD4-_mean_min_dist','CD20+_min_dist','CD3+CD4+_min_dist','CD3+CD4-_min_dist','Minor_axis_length','CD20+_object_number','CD3+CD4+_object_number','CD3+CD4-_object_number','Perim_circ_ratio','Perimeter','Pixel_size','Prob','Roi_num','Solidity']
    feature_list_auto=['Abs_coords','Area','Bbox_coords','Case','Centroid_x','Centroid_y','Circularity','Class_id','Convex_area','Convex_perimeter','Eccentricity','Equivalent_diameter','Major_axis_length', 'Major_minor_axis_ratio','BDCA2+_mean_min_dist','CD11c+_mean_min_dist','CD20+_mean_min_dist','CD3+CD4+_mean_min_dist','CD3+CD4-_mean_min_dist','BDCA2+_min_dist','CD11c+_min_dist','CD20+_min_dist','CD3+CD4+_min_dist','CD3+CD4-_min_dist','Minor_axis_length','BDCA2+_object_number','CD11c+_object_number','CD20+_object_number','CD3+CD4+_object_number','CD3+CD4-_object_number','Perim_circ_ratio','Perimeter','Pixel_size','Prob','Roi_num','Solidity']
    #feature_list_auto=['Area','Case','Centroid_x','Centroid_y','Circularity','Class_id','Convex_area','Convex_perimeter','DC','Eccentricity','Equivalent_diameter','Major_axis_length', 'Major_minor_axis_ratio','DC_mean_min_dist','CD3+CD4+_mean_min_dist','CD3+CD4-_mean_min_dist','DC_min_dist','CD3+CD4+_min_dist','CD3+CD4-_min_dist','Minor_axis_length','DC_object_number','CD3+CD4+_object_number','CD3+CD4-_object_number','Perim_circ_ratio','Perimeter','Pixel_size','Prob','Roi_num','Solidity']
    #feature_list_auto=['Area','Case','Circularity','Class_id','Convex_area','Convex_perimeter','DC','Eccentricity','Equivalent_diameter','Major_axis_length', 'Major_minor_axis_ratio','DC_mean_min_dist','Tw_mean_min_dist','Ta_mean_min_dist','DC_min_dist','Tw_min_dist','Ta_min_dist','Minor_axis_length','DC_object_number','Tw_object_number','Ta_object_number','Perim_circ_ratio','Perimeter','Pixel_size','Prob','Roi_num','Solidity']

    import pandas as pd
    auto_feats=pd.DataFrame.from_dict(auto_props)
    auto_feats=auto_feats[feature_list_auto]
    auto_feats.to_csv(os.path.join('../../',args.trainlogdir,'analysis','Human_FFPE_auto_feats_MR.csv'))
    print('done with feat')


    writedir=os.path.join('../../',args.trainlogdir,'analysis','new_analysis')
    if os.path.exists(writedir):
        shutil.rmtree(writedir)
        os.makedirs(writedir)
    else:
        os.makedirs(writedir)

    #sub_list_auto=['Circularity','Major_minor_axis_ratio',
    #'Perim_circ_ratio','Equivalent_diameter','Area','Perimeter',
    #'Eccentricity','Major_axis_length','Minor_axis_length','Solidity','Convex_area','Convex_perimeter',
    #'cd20_object_number','bdca2_object_number','cd11c_object_number']
    sub_list_auto=['Area','Circularity','Convex_area','Convex_perimeter','Eccentricity','Equivalent_diameter','Major_axis_length','Major_minor_axis_ratio','Minor_axis_length','Perim_circ_ratio','Perimeter','Solidity']
    prob=0
    mf=pd.DataFrame.from_csv(os.path.join('../../',args.trainlogdir,'analysis','Human_FFPE_auto_feats_MR.csv'))
    print(mf.shape)
    num_double=mf[(mf.Class_id=='CD3+CD4+')&(mf.Prob>prob)].shape[0]
    print('number of CD3+CD4+: ',num_double)
    mean_cd20dist=mf[(mf.Class_id=='CD3+CD4+')&(mf.Prob>prob)]['CD20+_min_dist'].mean()
    print(mean_cd20dist)
    mean_bdca2dist=mf[(mf.Class_id=='CD3+CD4+')&(mf.Prob>prob)]['BDCA2+_min_dist'].mean()
    print(mean_bdca2dist)
    mean_cd11cdist=mf[(mf.Class_id=='CD3+CD4+')&(mf.Prob>prob)]['CD11c+_min_dist'].mean()
    print(mean_cd11cdist)
    num_single=mf[(mf.Class_id=='CD3+CD4-')&(mf.Prob>prob)].shape[0]
    print('number of CD3+CD4-: ',num_single)
    mean_CD20dist=mf[(mf.Class_id=='CD3+CD4-')&(mf.Prob>prob)]['CD20+_min_dist'].mean()
    print(mean_CD20dist)
    mean_BDCA2dist=mf[(mf.Class_id=='CD3+CD4-')&(mf.Prob>prob)]['BDCA2+_min_dist'].mean()
    print(mean_BDCA2dist)
    mean_CD11cdist=mf[(mf.Class_id=='CD3+CD4-')&(mf.Prob>prob)]['CD11c+_min_dist'].mean()
    print(mean_CD11cdist)
    a=0
    b=0
    from scipy.stats import ttest_ind
    for dist_measure in ['CD20+_min_dist','BDCA2+_min_dist','CD11c+_min_dist']:
        print(dist_measure)
        a=mf[(mf.Class_id=='CD3+CD4+')&(mf.Prob>prob)][dist_measure]
        b=mf[(mf.Class_id=='CD3+CD4-')&(mf.Prob>prob)][dist_measure]
        (tstat,p)=ttest_ind(a.dropna(), b.dropna(), axis=0, equal_var=True)
        print('ttest')
        print(tstat)
        print(p)
        print('ttest')
        (tstat,p)=ttest_ind(a.dropna(), b.dropna(), axis=0, equal_var=False)
        print(tstat)
        print(p)
    a=0; b=0; c=0
    for case in mf['Case'].unique():
        for roi_num in mf[(mf.Case==case)]['Roi_num'].unique():
                a=a+mf[(mf.Case==case)&(mf.Roi_num==roi_num)]['CD20+_object_number'].iloc[0]
                b=b+mf[(mf.Case==case)&(mf.Roi_num==roi_num)]['BDCA2+_object_number'].iloc[0]
                c=c+mf[(mf.Case==case)&(mf.Roi_num==roi_num)]['CD11c+_object_number'].iloc[0]
    print('number of cd20: ',a)
    print('number of bdca2: ',b)
    print('number of cd11c: ',c)
    #scaler=MinMaxScaler()
    #scaler.fit(mf[sub_list_auto])
    #mf[sub_list_auto]=scaler.transform(mf[sub_list_auto])
    '''
    def plot_feats_auto(df,sub_list,dirname,prob,partition_vars,pvarmy):
        for feat in sub_list:
            f, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,sharey=True,sharex=True,figsize=(10,15))
            ax1.set_title('cd20 CD3+CD4+')
            ax1.plot(
                    df[(df.Class_id=='CD3+CD4+')&(df.Prob>prob)][partition_vars[0]],
                    df[(df.Class_id=='CD3+CD4+')&(df.Prob>prob)][feat],'r.')
            ax2.set_title('cd20 CD3+CD4-')
            ax2.plot(
                    df[(df.Class_id=='CD3+CD4-')&(df.Prob>prob)][partition_vars[0]],
                    df[(df.Class_id=='CD3+CD4-')&(df.Prob>prob)][feat],'r.')
            ax3.set_title('bdca2 CD3+CD4+')
            ax3.plot(
                    df[(df.Class_id=='CD3+CD4+')&(df.Prob>prob)][partition_vars[1]],
                    df[(df.Class_id=='CD3+CD4+')&(df.Prob>prob)][feat],'r.')
            ax4.set_title('bdca2 CD3+CD4-')
            ax4.plot(
                    df[(df.Class_id=='CD3+CD4-')&(df.Prob>prob)][partition_vars[1]],
                    df[(df.Class_id=='CD3+CD4-')&(df.Prob>prob)][feat],'r.')
            ax5.set_title('cd11c CD3+CD4+')
            ax5.plot(
                    df[(df.Class_id=='CD3+CD4+')&(df.Prob>prob)][partition_vars[2]],
                    df[(df.Class_id=='CD3+CD4+')&(df.Prob>prob)][feat],'r.')
            ax6.set_title('cd11c CD3+CD4-')
            ax6.plot(
                    df[(df.Class_id=='CD3+CD4-')&(df.Prob>prob)][partition_vars[2]],
                    df[(df.Class_id=='CD3+CD4-')&(df.Prob>prob)][feat],'r.')
            f.savefig(os.path.join(writedir,'feats_'+pvarmy+'_'+feat+'.png'))

    partition_vars=['CD20_min_dist','BDCA2_min_dist','CD11c_min_dist']
    plot_feats_auto(mf,sub_list_auto,writedir,prob,partition_vars,'Min_dist')
    partition_vars=['CD20_mean_min_dist','BDCA2_mean_min_dist','CD11c_mean_min_dist']
    plot_feats_auto(mf,sub_list_auto,writedir,prob,partition_vars,'Mean_min_dist')

    def plot_auc(aucs,name1,name2,rng,folder,partition_variable):
        f,axes  = plt.subplots(2,2,sharey=True,sharex=True,figsize=(10,10))
        for i in range(len(axes)):
            for j in range(len(axes[i])):
                k=i*2+j
                mean=np.array([np.mean(x) for x in aucs[:,k,0]])
                stdev=np.array([np.std(x) for x in aucs[:,k,0]])
                axes[i][j].plot(rng,mean)
                axes[i][j].plot(rng,mean-aucs[:,k,1])
                axes[i][j].plot(rng,mean+aucs[:,k,1])
                axes[i][j].plot(rng,0.5*np.ones(len(rng)))
                axes[i][j].plot(rng,0.6*np.ones(len(rng)))
        axes[0][0].set_title(name1+'_low vs '+name1+'_high')
        axes[0][1].set_title(name2+'_low vs '+name2+'_high')
        axes[1][0].set_title(name1+'_low vs '+name2+'_low')
        axes[1][1].set_title(name1+'_high vs '+name2+'_high')
        axes[1][1].set_ylim([0,1.0])
        f.savefig(os.path.join(folder,'without_dist_'+partition_variable+'_'+name1+'_'+name2+'.png'))

    def min_dist_analysis_auto(df,sub_list,folder,rng,kernel,testsize,partition_variable,prob):

        ylist1=an.test_mindist_low(
                df[(df.Class_id=='CD3+CD4+')&(df.Prob>prob)][sub_list+[partition_variable]].dropna(axis=0,how='any',subset=[partition_variable]).copy(),
                df[(df.Class_id=='CD3+CD4-')&(df.Prob>prob)][sub_list+[partition_variable]].dropna(axis=0,how='any',subset=[partition_variable]).copy(),
                rng,
                kernel,
                testsize,
                partition_variable)

        plot_auc(ylist1,'CD3+CD4+','CD3+CD4-',rng,folder,partition_variable)

    rng=0.2*np.array(list(range(1,100)))

    #for p in range(10):
    #    prob=p*0.1
    #    min_dist_analysis_auto(af,sub_list_auto,'auto_feats',rng,prob)
    kernel='rbf'
    testsize=0.2
    pvars=['CD20+_min_dist','BDCA2+_min_dist','CD11c+_min_dist','CD20+_mean_min_dist','BDCA2+_mean_min_dist','CD11c+_mean_min_dist']
    for partition_variable in pvars:
            min_dist_analysis_auto(mf,sub_list_auto,writedir,rng,kernel,testsize,partition_variable,prob)

    print('done with min dist analysis')

    # classify by all features including dist, need to normalize dist

    def with_dist_analysis(df,sub_list,folder,partition_variable,prob):

        sub_list=sub_list+[partition_variable]

        scaler=MinMaxScaler()
        scaler.fit(df[sub_list])
        df[sub_list]=scaler.transform(df[sub_list])

        auclist=an.try_classify(
                df[(df.Class_id=='CD3+CD4+')&(df.Prob>prob)][sub_list].dropna(axis=0,how='any',subset=[partition_variable]).values,
                df[(df.Class_id=='CD3+CD4-')&(df.Prob>prob)][sub_list].dropna(axis=0,how='any',subset=[partition_variable]).values,
                'rbf',0.2)
        plt.figure()
        plt.plot(auclist[0])
        plt.savefig(os.path.join(folder,'with_dist_'+partition_variable+'_CD3+CD4+_vs_CD3+CD4-.png'))

    # for auto features
    mf=pd.DataFrame.from_csv(os.path.join(maindir,'feats.csv'))
    mf=mf.dropna(axis=0,how='any')
    for partition_variable in pvars:
            with_dist_analysis(mf,sub_list_auto,writedir,partition_variable,prob)

    # variance analysis

    print('done')
    '''
if __name__=='__main__':
    main()
