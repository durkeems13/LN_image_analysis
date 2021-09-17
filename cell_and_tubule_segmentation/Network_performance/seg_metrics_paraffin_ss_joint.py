import pandas as pd
import numpy as np
from glob import glob
import warnings
import os
import argparse
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--csv_name",type=str,default="cdm3_analysis/MR_transfer.csv",help = '')
args,unparsed = parser.parse_known_args()
classes=['CD3+CD4+','CD3+CD4-','CD20+','BDCA2+','CD11c+'] 
csvfilename = args.csv_name
print()
iou_cutoff=0.25 #float(filename[-8:-4])
print('analysis for detection threshold of '+"{:.2f}".format(iou_cutoff)+' intersection over union')
df=pd.read_csv(csvfilename)

a=[df[df.Class_id==y][df.iou>iou_cutoff]['iou'].mean() for y in classes]
meanlist=["{0:.2f}".format(x) for x in a]
b=[df[df.Class_id==y][df.iou>iou_cutoff]['iou'].std() for y in classes]
stdevlist=["{0:.2f}".format(x) for x in b]
print()
print('per class iou with standard deviation')
print(' '.join([' '.join([x,y,'+/-',z]) for (x,y,z) in zip(classes,meanlist,stdevlist)]))

a=df[df.iou>iou_cutoff]['iou'].mean()
b=df[df.iou>iou_cutoff]['iou'].std()
print('average iou across classes with standard deviation')
print("{0:.2f}".format(a)+' +/- '+"{0:.2f}".format(b))

tp=[df[df.Class_id==y][df.Detection=='tp']['iou'].shape[0] for y in classes]
fn=[df[df.Class_id==y][df.Detection=='fn']['iou'].shape[0] for y in classes]
c=[x/(x+y) for (x,y) in zip(tp,fn)]
TP = df[df.Detection=='tp'].shape[0]
FN = df[df.Detection=='fn'].shape[0]
C = TP/(TP+FN)

print()
print('per class sensitivty')
sens_list=["{0:.2f}".format(x) for x in c]
print(' '.join([' '.join([x,y]) for (x,y) in zip(classes,sens_list)]))
print('average sensitivity across classes')
print("{0:.2f}".format(C))

tn=[tp[4]+tp[3]+tp[2]+tp[1]+fn[4]+fn[3]+fn[2]+fn[1],tp[4]+tp[3]+tp[2]+tp[0]+fn[4]+fn[3]+fn[2]+fn[0],tp[4]+tp[3]+tp[1]+tp[0]+fn[4]+fn[3]+fn[1]+fn[0],tp[4]+tp[2]+tp[1]+tp[0]+fn[4]+fn[2]+fn[1]+fn[0],tp[3]+tp[2]+tp[1]+tp[0]+fn[3]+fn[2]+fn[1]+fn[0]]
fp=[df[df.Class_id==y][df.Detection=='fp']['iou'].shape[0] for y in classes]
FP = df[df.Detection=='fp'].shape[0]
c=[x/(x+y) for (x,y) in zip(tn,fp)]


print()
print('per class specificity')
spec_list=["{0:.2f}".format(x) for x in c]
print(' '.join([' '.join([x,y]) for (x,y) in zip(classes,spec_list)]))
print('average specificity across classes')
print("{0:.2f}".format(np.mean(c)))

c = [x/(x+y) for (x,y) in zip(tp,fp)]
C = TP/(TP+FP)
print()
print('per class precision')
spec_list=["{0:.2f}".format(x) for x in c]
print(' '.join([' '.join([x,y]) for (x,y) in zip(classes,spec_list)]))
print('average precision across classes')
print("{0:.2f}".format(C))

print()
print('number of manually segmented cells by class')
mandet = ['tp','fn']
shape_list=[df[df.Class_id==x][df.Detection.isin(mandet)].shape[0] for x in classes]
print(' '.join([' '.join([x,str(y)]) for (x,y) in zip(classes,shape_list)]))
print('total number of manually segmented cells')
print(df.shape[0])

print()
print('iou for all manual cells in set with no detection threshold')

a=[df[df.Class_id==y]['iou'].mean() for y in classes]
meanlist=["{0:.2f}".format(x) for x in a]
b=[df[df.Class_id==y]['iou'].std() for y in classes]
stdevlist=["{0:.2f}".format(x) for x in b]
print('per class iou with standard deviation')
print(' '.join([' '.join([x,y,'+/-',z]) for (x,y,z) in zip(classes,meanlist,stdevlist)]))


#a=df['iou'].mean()
#b=df['iou'].std()
#print('average iou across classes with standard deviation')
#print("{0:.2f}".format(a)+' +/- '+"{0:.2f}".format(b))
