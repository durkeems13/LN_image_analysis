from glob import glob
from random import shuffle
import pickle as pkl
import argparse
import os

# use this to create text file to index work for the parallel sbatch job
# output to text file with 'python indices_tofile.py > out.txt'
def main():

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--folder",
        type=str,
        default='../../joint_DC_and_lymphocyte_preds/processing/feature_stage_pkls',
        help=""
    )

    args, unparsed = parser.parse_known_args()
    strings=[]
    for x in glob(os.path.join(args.folder,'*')):
        thing=pkl.load(open(x,'rb'))
        try:
            for i,y in enumerate(thing[1]):
                strings.append('{:04d}'.format(i)+' '+x.split('/')[-1])
        except:
            continue
    #shuffle the work for the batch job
    shuffle(strings)
    '''
    with open('../../joint_DC_and_lymphocyte_preds/processing/indices.txt','w') as f:
        for x in strings:
            #print(x)
            f.write(x)
    '''
    for x in strings:
        print(x)
if __name__=='__main__':
    main()
