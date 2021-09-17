import os,csv,pickle,sys,pprint,time,operator,itertools,copy,argparse,shutil
import pickle as pkl
from matplotlib import pyplot as plt
from itertools import chain
from random import shuffle
import importlib as imp
import numpy as np
# use imp.reload(an) to reload analysis
import pandas as pd
import operator
import cv2
import tqdm
from joblib import Parallel, delayed
import multiprocessing
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc
from tifffile import imread,imsave
from pycocotools import mask as pycocomask
from skimage.measure import label, regionprops, find_contours
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from imagej_tiff_meta import TiffWriter
from shapely.geometry import Polygon
import sys
sys.path.append('../experiments/CDM_ResNet_validation_FFPE_ss_DC')
import eval

#def close_contour(arr):
    

def fill_contours(arr):
    return np.maximum.accumulate(arr, 1) &\
           np.maximum.accumulate(arr[:, ::-1], 1)[:, ::-1] &\
           np.maximum.accumulate(arr[::-1, :], 0)[::-1, :] &\
           np.maximum.accumulate(arr, 0)

def main():

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--pkls_read",
        type=str,
        default='../analysis/final_network_perf/processing/combined_inference_pkls',
        help=""
    )
    parser.add_argument(
        "--gt_pkls",
        type=str,
        default='../../mrcnn_human_ffpe_ss_DC/data/manual_pkls/Human_paraffin_single_stain_GT',
        help=""
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default='LuN_final_with_spec.csv',
        help=""
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help=""
    )

    args, unparsed = parser.parse_known_args()
    pred_fldr=args.pkls_read
    #print('processing ',args.csv_name)
    #pred_pkls = os.listdir(pred_fldr)
    print(args.threshold,' THRESHOLD')
    pred_pkls = os.listdir(pred_fldr)
    pred_pkls.sort()
    gtfolder=args.gt_pkls
    gt_pkls = os.listdir(pred_fldr)
    gt_pkls.sort()
    pred_check = [x.split('.')[0] for x in pred_pkls]
    gt_check = [x.split('.')[0] for x in gt_pkls]
    cases = list(set(pred_check).intersection(gt_check))
    '''
    csvpath = args.csv_name.split('/')
    if len(csvpath) > 1:
        csv_path = '/'.join(csvpath[:-2])
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
    '''
    #ths = {'CD3+CD4+':0.5,'CD3+CD4-':0.4,'CD20+':0.5,'BDCA2+':0.8,'CD11c+':0.3}
    auto_segs=[]
    manual_segs=[]
    cases.sort()
    #cases = cases[:2]
    for i,case in enumerate(cases):
        case_gt_name = case+'.tif.pkl'
        case_pred_name = case+'.pkl'
        gt_pkl = pkl.load(open(os.path.join(gtfolder,case_gt_name),'rb'))
        gt_dict = []
        if len(gt_pkl):
            for j in range(len(gt_pkl)):
                gt_dict.append({'Casename':case,'Class_id':gt_pkl[j]['Class_id'],'Coords':gt_pkl[j]['Coords']})
                #gt_dict.append({'Casename':case,'Class_id':gt_pkl[2][j],'Coords':gt_pkl[3][j]})
        try:
            pred_pkl = pkl.load(open(os.path.join(pred_fldr,case_pred_name),'rb'))
        except:
            #print('Database error at: ',case_pred_name)
            pred_pkl = []
        pred_dict = []
        if len(pred_pkl):
            for j in range(len(pred_pkl)):
                #pred_dict.append({'Casename':case,'Class_id':pred_pkl[j].class_id,'Coords':pred_pkl[j].mask,'Score':pred_pkl[j].score})
                pred_dict.append({'Casename':case,'Class_id':pred_pkl[j]['class_id'],'Coords':pred_pkl[j]['mask'],'Score':pred_pkl[j]['score']})
        else:
            pred_dict.append({'Casename':'','Class_id':10,'Coords':[],'Score':0})
        auto_segs.append(pred_dict)
        manual_segs.append(gt_dict)
    auto_segs2 = []
    manual_segs2 = []
    for im in auto_segs:
        im = [x for x in im if x['Score']>args.threshold]
        #im = [x for x in im if 'S08-2012' not in x['Casename']]
        auto_segs2.append(im)
    for im in manual_segs:
        #im = [x for x in im if 'S08-2012' not in x['Casename']]
        manual_segs2.append(im)
    all_cells = pd.DataFrame()
    for i,im in enumerate(manual_segs2):
        if len(im) > 0:
            imname = im[0]['Casename']
        else:
            continue
        whichi = []
        thisi = []
        for ii,autoim in enumerate(auto_segs2):
            if len(autoim) > 0:
                testcase = autoim[0]['Casename']
                if testcase == imname:
                    whichi.append(ii)
            else:
                continue
        if len(whichi) >= 1:
            thisi = whichi[0]
        manual_list_save = []
        auto_list_save = []
        auto_pop_list = []
        pred_im2 =[]
        if thisi:
            pred_im2 = auto_segs2[thisi]
            '''
            # following loop selects class-specific thresholds
            for pcell in pred_im:
                cid = pcell['Class_id']
                if cid == 1:
                    classname='CD20+'
                elif cid == 2:
                    classname='CD3+CD4-'
                elif cid == 3:
                    classname='CD3+CD4+'
                elif cid == 4:
                    classname='BDCA2+'
                elif cid == 5:
                    classname='CD11c+'
                if pcell['Score'] >= ths[classname]:
                    pred_im2.append(pcell)
                '''
        else:
            pred_im2 = []
        #if len(pred_im)> 0 and len(im) >0:
        #    print(pred_im[0]['Casename'],im[0]['Casename'])
        for obj in im:
            pts = obj['Coords']
            pts = [(x,y) for (x,y) in zip(pts[0],pts[1])]
            polypts = []
            for pt in pts:
                polypts.append(np.uint16(pt[1]))
                polypts.append(np.uint16(pt[0]))
            if len(polypts) < 6:
                continue
            classid = obj['Class_id']
            ovlist = []
            mmask = np.zeros([1024,1024])
            re = pycocomask.frPyObjects([polypts],1024,1024)
            mmask = pycocomask.decode(re)
            mmask = mmask[:,:,0]
            for j,acell in enumerate(pred_im2):
                amask = acell['Coords']*5
                mask = amask+mmask
                intersection = np.sum(mask==6)
                union = np.sum(mask > 0)
                iou = intersection/union
                ovlist.append(iou)
            if ovlist:
                maxindex,maxiou = max(enumerate(ovlist),key=operator.itemgetter(1))
                maxclass = pred_im2[maxindex]['Class_id']
                maxscore = pred_im2[maxindex]['Score']
                classname=classid
                if classname=='CD20+':
                    classid=1
                elif classname=='CD3+CD4-':
                    classid=2
                elif classname=='CD3+CD4+':
                    classid=3
                elif classname=='BDCA2+':
                    classid=4
                elif classname=='CD11c+':
                    classid=5
                if maxiou > 0.25:
                    if maxclass == classid:
                        obj['Detection']='tp'
                        obj['iou']=maxiou
                        obj['Class_id']=classname
                        obj.pop('Coords',None)
                        auto_cell = {'Casename':obj['Casename'],'Class_id':classname,'Detection':'atp','iou':maxiou,'Score':maxscore}
                        auto_list_save.append(auto_cell)
                        manual_list_save.append(obj)
                        auto_pop_list.append(maxindex)
                    else:
                        obj['Detection']='fn'
                        obj['iou']=maxiou
                        obj['Class_id']=classname
                        obj['Score']=0.0
                        obj.pop('Coords',None)
                        manual_list_save.append(obj)
                else:
                    obj['Detection']='fn'
                    obj['iou']=maxiou
                    obj['Class_id']=classname
                    obj['Score'] = 0.0
                    obj.pop('Coords',None)
                    manual_list_save.append(obj)
        fps = [x for i,x in enumerate(pred_im2) if i not in auto_pop_list]        
        for fp in fps:
            casename = fp['Casename']
            if fp['Class_id']==1:
                autoclassid='CD20+'
            elif fp['Class_id']==2:
                autoclassid='CD3+CD4-'
            elif fp['Class_id']==3:
                autoclassid='CD3+CD4+'
            elif fp['Class_id']==4:
                autoclassid='BDCA2+'
            elif fp['Class_id']==5:
                autoclassid='CD11c+'
            classid = autoclassid
            iou = 0
            auto_cell = {'Casename':casename,'Class_id':classid,'Detection':'fp','iou':iou,'Score':fp['Score']}
            auto_list_save.append(auto_cell)
        man_df = pd.DataFrame(manual_list_save)
        auto_df = pd.DataFrame(auto_list_save)
        all_cells = pd.concat([all_cells,man_df,auto_df])
        #if len(im):
        #    print(im[0]['Casename'],'DONE')
    all_cells.to_csv(args.csv_name)

if __name__ == '__main__':
    main()
