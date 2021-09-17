# use this script with feat.sbatch to compute distance relationships and shape
# features of cells

import os,csv,pickle,sys,pprint,time,operator,itertools,copy,argparse,shutil
import pickle as pkl
from matplotlib import pyplot as plt
from itertools import chain
from random import shuffle
import importlib as imp
import numpy as np
# use imp.reload(an) to reload analysis

from tifffile import imread,imsave
from scipy.spatial.distance import cdist
from skimage.measure import label,regionprops

def compute_cell_features(class_index,cell_index,points,props,classes,scores,class_names,pixelsize,roinum,casename):
    
    newcell={}
    valid_indices=list(range(len(props)))
    valid_indices.pop(cell_index)
    cell=props[cell_index]
    props=np.delete(props,cell_index)
    cell_points=points[cell_index]
    cell_points = [(x,y) for i,(x,y) in enumerate(zip(cell_points[0],cell_points[1]))]
    points=np.delete(points,cell_index,axis=0)
    classes=np.delete(classes,cell_index)
    #print('scores',scores)
    for class_index_other in range(5):#3 if FF
        distmatlist_min=[] 
        distmatlist_mean=[]
        cell_name=class_names[class_index_other]#relative to this
        class_name=class_names[class_index_other]
        cell_props_new=props[classes==(class_index_other+1)]
        points_other=points[classes==(class_index_other+1)]
        '''
        print(cell['coords'])
        print('points')
        print(cell_points[0],cell_points[1])
        #print('cell') 
        #print(cell)
        #print('coords')
        #print(cell.coords)
        '''
# cellcoords is a stack of coordinates of the imported cell, meanpoint is mean coordinate 
        cellcoords=np.stack([cell.coords[:,0]+np.min(cell_points[0]),cell.coords[:,1]+np.min(cell_points[1])],axis=1)
        meanpoint_cellcoords=cellcoords.mean(axis=0)[np.newaxis,:]
        for iii,(dc,dcpoints) in enumerate(zip(cell_props_new,points_other)):
            #dc and dc points are for next class
            dccoords=np.stack([dc.coords[:,0]+np.min(dcpoints[0]),dc.coords[:,1]+np.min(dcpoints[1])],axis=1)
            meanpoint_dccoords=dccoords.mean(axis=0)[np.newaxis,:]
            # to speed up processing don't bother to compute distances to cells
            # mean mean coords greater than 1000 pixels away
            dist_mean_points=cdist(meanpoint_cellcoords,meanpoint_dccoords)[0][0]
           
            if dist_mean_points < 2000:
                distmatlist_min.append(cdist(cellcoords,dccoords).min())
                distmatlist_mean.append(cdist(cellcoords,dccoords).mean())
            else:
                distmatlist_min.append(dist_mean_points)
                distmatlist_mean.append(dist_mean_points)
            
        if len(distmatlist_min):
            newcell[cell_name+'_min_dist']=min(distmatlist_min)*pixelsize
            newcell[cell_name+'_mean_min_dist']=min(distmatlist_mean)*pixelsize
            if class_name==cell_name:
                newcell[cell_name+'_object_number']=len(distmatlist_min)+1
            else:
                newcell[cell_name+'_object_number']=len(distmatlist_min)

        else:
            newcell[cell_name+'_min_dist']=np.nan
            newcell[cell_name+'_mean_min_dist']=np.nan
            if class_name==cell_name:
                newcell[cell_name+'_object_number']=1
            else:
                newcell[cell_name+'_object_number']=0
    CHprop = regionprops(label(cell.convex_image,connectivity=2))
    newcell['Abs_coords'] = cell_points
    newcell['Centroid_x'] = cell.centroid[0]+np.min(cell_points[0])#np.sum(list(zip(*newcell['Abs_coords']))[0])/len(newcell['Abs_coords'])
    newcell['Centroid_y'] = cell.centroid[1]+np.min(cell_points[1])#np.sum(list(zip(*newcell['Abs_coords']))[1])/len(newcell['Abs_coords'])
    newcell['Bbox_coords'] = cell.bbox
    newcell['Coords'] = cell.coords
    #newcell['coords'] = cell.coords 
    newcell['Area']=cell.area*pixelsize*pixelsize
    newcell['Case']=casename
    newcell['Perimeter']=cell.perimeter*pixelsize
    newcell['Circularity']=np.power(newcell['Perimeter'],2)/(4*np.pi*newcell['Area'])
    newcell['Class_id']=class_names[class_index]
    newcell['Convex_area']=cell.convex_area*pixelsize*pixelsize
    newcell['Convex_perimeter']=CHprop[0].perimeter*pixelsize
    newcell['DC'] = casename[0:2]
    newcell['Major_axis_length']=cell.major_axis_length*pixelsize
    newcell['Minor_axis_length']=cell.minor_axis_length*pixelsize
    newcell['Eccentricity'] = np.sqrt(newcell['Major_axis_length']**2+newcell['Minor_axis_length']**2)/newcell['Major_axis_length']
    newcell['Equivalent_diameter']=cell.equivalent_diameter*pixelsize
    try:
        newcell['Major_minor_axis_ratio']=newcell['Major_axis_length']/newcell['Minor_axis_length']
    except ZeroDivisionError:
        newcell['Major_minor_axis_ratio']=0
    newcell['Perim_circ_ratio']=newcell['Perimeter']/newcell['Circularity']
    newcell['Pixel_size']=pixelsize
    newcell['Prob']=scores[cell_index]
    newcell['Roi_num']=roinum
    newcell['Solidity'] = cell.solidity

    return newcell

def main():

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--index",
        type=str,
        default='1:5',
        help=""
    )
    parser.add_argument(
        "--trainlogdir",
        type=str,
        default='human_paraffin_ss_all_classes_per_im',
        help=""
    )

    args, unparsed = parser.parse_known_args()
    index=args.index
    index=index.split(':')
    index=index[-1]
    index=int(index)-1
    print(index)

    with open(os.path.join('../../',args.trainlogdir,'processing','indices.txt')) as f:
        content=f.readlines()
    content=[x.strip() for x in content]
    
    print(index)
    
    content=content[index]
    content=content.split(' ')
    print(content)
    print(content[1])


    cellindex=int(content[0])
    pklload=pkl.load(open(os.path.join('../../',args.trainlogdir,'processing','feature_stage_pkls',content[1]),'rb'))
    myinput=list(pklload[1][cellindex])+pklload[0]
    # print(myinput)
    write_dir=os.path.join('../../',args.trainlogdir,'processing','output_feats')
    if not(os.path.exists(write_dir)):
        os.makedirs(write_dir)
    
    output=compute_cell_features(*myinput)
    #print('output',output)
    file_name=content[1]+'_'+str(cellindex)
    write_file=os.path.join(write_dir,file_name)
    pkl.dump(output,open(write_file,'wb'))
    print('done')

if __name__ == '__main__':
    main()
