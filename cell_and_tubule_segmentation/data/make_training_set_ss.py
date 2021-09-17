# use this script to generate training set with train/validation
# partition for use with the coco api format
# import standard libraries required
import os,csv,sys,pprint,time,operator,shutil,json,argparse
from matplotlib import pyplot as plt
import numpy as np
import importlib as imp
import pickle as pkl
from random import shuffle
from itertools import chain
from glob import glob
# other libraries
from tifffile import imread,imsave
from imagej_tiff_meta import TiffFile
import xml.etree.ElementTree as ET

sys.path.append('../code/lib')
from pycocotools import mask

# use imp.reload(an) to reload analysis
#from sklearn.preprocessing import MinMaxScaler
#from scipy.stats.mstats import zscore
#from joblib import Parallel, delayed
#import multiprocessing

# parse overlays from tiffiles to get out per channel information
def process_overlays(overlays):

    # only get info with multi-coordinates key
    '''
    bboxes=[overlays[i] for i in range(len(overlays)) if overlays[i]['roi_type']==1]
    bboxes_points=[np.array([[x['left'],x['top']],[x['left'],x['bottom']],
        [x['right'],x['bottom']],[x['right'],x['top']]]) for x in bboxes]
    bboxes_channels=[x['position']-1 for x in bboxes]
    '''
    #[print(overlays[i]['roi_type']) for i in range(len(overlays))]
    overlays=[x for x in overlays if x['roi_type']!=1]
    p1=lambda x,key : [x[i][key] for i in range(len(x))
            if 'multi_coordinates' in x[i].keys()]
    kys=['position','left','top','multi_coordinates','name']
    overlays=[p1(overlays,key) for key in kys]
    
    # eliminate all the Nones
    p2=lambda x,y : [a for (a,b) in zip(x,y) if y != None]

    overlays=[p2(x,overlays[3]) for x in overlays]

    # remove list wrapper from points
    overlays[3]=[x[0] for x in overlays[3]]

    new_overlay_points=[]
    new_overlay_channels=[]
    new_overlay_names=[]
    for (c,left,top,x,name) in zip(*overlays):
        if x.ndim ==2:
            x[:,0] += left
            x[:,1] += top
            new_overlay_points.append(x)
            #new_overlay_channels.append(c-1)
            new_overlay_channels.append(c)
            new_overlay_names.append(name)
    overlay_probs=[]
    for x,y in zip(new_overlay_names,new_overlay_channels):
        overlay_probs.append(1.0)
    '''
    # get probabilities out of names
    overlay_probs=[]
    for x,y in zip(new_overlay_names,new_overlay_channels):
        xx=x.split('-')[0]
        #print(xx)
        if xx=='':
            #print(x)
            overlay_probs.append(1.0)
        elif xx[1]=='.':
            print(xx)
            print(y)
            #comment following 5 lines to keep all prob values
            xx_val = float(xx)
            if xx_val < 0.6:
               xx_val = 0
            #else:
               #xx_val = 1.0
            overlay_probs.append(xx_val)
        else:
            overlay_probs.append(1.0)
    print(np.unique(overlay_probs))
    '''
    points_by_channel=[]
    for i in [0,1,2]:        
        points_by_channel.append([(x,prob) for x,y,prob in zip(new_overlay_points,new_overlay_channels,overlay_probs)
        if y==i])
    return points_by_channel

# create annotations in the coco dataset format
def add_annotations(annot_index,points,image_id,imh,imw):
    annotations=[]
    for i,point_set in enumerate(points):
        for polygon,prob in point_set:
            if prob > 0.5:
                polygon=polygon.flatten()
                if polygon.shape[0] > 4: 
                    re = mask.frPyObjects([polygon],imh,imw)
                    re=re[0]
                    area = mask.area( re )
                    if area > 20:
                        bbox = mask.toBbox( re )
                        area=int(area)
                        bbox=list(bbox)
                        re['counts'] = re['counts'].decode('utf-8')
                        polygon=list(polygon)
                        polygon=[float(x) for x in polygon]
                        annotation_info={'segmentation':[polygon],
                                        'image_id':image_id,
                                        'category_id':i+1,
                                        'id':annot_index,
                                        'iscrowd':0,
                                        'area':area,
                                        'bbox':bbox,
                                        'score':prob
                        }
                        annotations.append(annotation_info)
                        annot_index=annot_index+1
    return annotations,annot_index

# add image info
def add_images(roipath,image_id,rootfolder,save_folder,imh,imw,means):
    imagename=roipath.split('/')[-1]
    newpath=os.path.join(rootfolder,save_folder,imagename)
    shutil.copy(roipath,newpath) 
    image_info={'path':imagename,'file_name':imagename,'id':image_id,'height':imh,'width':imw
    }
    return image_info

# loop over rois withint the training or val set and write
# label and image info
def save_set(myset,save_folder,annot_index,rootfolder,image_id,means):
    annotations=[]
    images=[]
    os.makedirs(os.path.join(rootfolder,save_folder))
    for roipath in myset:
        print(roipath)
        t=TiffFile(roipath)
        if 'parsed_overlays' in t.pages[0].imagej_tags.keys():
            overlays=t.pages[0].imagej_tags.parsed_overlays
        else:
            overlays = []
        t.close()

        points=process_overlays(overlays)
        imh=512 #1024
        imw=512 #1024
        annotation,annot_index=add_annotations(annot_index,points,image_id,imh,imw)

        annotations=annotations+annotation

        image_info=add_images(roipath,image_id,rootfolder,save_folder,imh,imw,means)
        images.append(image_info)
        image_id=image_id+1

    return images,annotations,annot_index,image_id

def main():

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--read",
        type=str,
        default="",
        help="images to read"
    )
    parser.add_argument(
        "--write",
        type=str,
        default="",
        help="directory to write to"
    )

    CMD_FLAGS, unparsed = parser.parse_known_args()
    
    root_dir=os.getcwd()
    read_dir=CMD_FLAGS.read
    savefldr=CMD_FLAGS.write
    
    roi_collection=list(glob(os.path.join(read_dir,'*.tif')))

    # shuffle the rois for randomness
    shuffle(roi_collection)

    savefldr=os.path.join(root_dir,savefldr)
    if os.path.exists(savefldr):
        shutil.rmtree(savefldr)
    os.makedirs(savefldr)
    os.makedirs(os.path.join(savefldr,'.log_db'))
    os.makedirs(os.path.join(savefldr,'.log_db',__file__.split('/')[-1][:-3]))
    os.makedirs(os.path.join(savefldr,'.log_db','deps'))
    #shutil.copytree(os.path.join(read_dir,'.log_db'),os.path.join(savefldr,'.log_db','deps',read_dir.split('/')[-1]))
    #random_trainset generation
    '''
    train_per=0.8
    val_per=0.1
    mid=int(len(roi_collection)*train_per)
    mid2=int(len(roi_collection)*(train_per+val_per))
    train_rois=roi_collection[:mid]
    val_rois=roi_collection[mid:mid2]
    test_rois=roi_collection[mid2:]
    '''
    #if image split already exists
    train_rois=list(glob(os.path.join(read_dir,'train','*.tif')))
    val_rois=list(glob(os.path.join(read_dir,'val','*.tif')))
    test_rois=list(glob(os.path.join(read_dir,'testset','*.tif')))                               
    means=[]
    for x in train_rois:
        im=imread(x)
        means.append(im.mean(axis=(1,2)))
    b=np.stack(means,axis=0).mean(axis=0)
    means=b[:,np.newaxis,np.newaxis]
    print(b)
    print('done')

    train_images,train_annotations,annot_index,image_id=save_set(train_rois,'train',0,savefldr,0,means)
    val_images,val_annotations,annot_index,image_id=save_set(val_rois,'val',annot_index,savefldr,image_id,means)
    test_images,test_annotations,annot_index,image_id=save_set(test_rois,'testset',annot_index,savefldr,image_id,means)
    cats=[{'name':'lymphocyte','id':1}]    
    #cats=[{'name':'bdca2','id':1},{'name':'cd11c','id':2}]
    #cats=[{'name':'cd20','id':1},{'name':'cd3+cd4-','id':2},{'name':cd3+cd4+','id':3}]
    #cats=[{'name':'SP','id':1},{'name':'DP','id':2}]

    savefldr_annotations=os.path.join(savefldr,'annotations')
    os.makedirs(savefldr_annotations)

    train_json_dict={'info':{'description':'my new train dataset','means':list(b)},
        'images':train_images,'annotations':train_annotations,'categories':cats}
    with open(os.path.join(savefldr_annotations,'train.json'),'w') as outfile:
        json.dump(train_json_dict,outfile)

    val_json_dict={'info':{'description':'my new val dataset'},
        'images':val_images,'annotations':val_annotations,'categories':cats}
    with open(os.path.join(savefldr_annotations,'val.json'),'w') as outfile:
        json.dump(val_json_dict,outfile)

    test_json_dict={'info':{'description':'my new test dataset'},
        'images':test_images,'annotations':test_annotations,'categories':cats}
    with open(os.path.join(savefldr_annotations,'testset.json'),'w') as outfile:
        json.dump(test_json_dict,outfile)

if __name__=="__main__": 
    main()
