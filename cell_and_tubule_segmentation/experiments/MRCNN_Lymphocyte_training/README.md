This folder contains the code necessary to train using tensorpack.

training logs are now stored in cdm3/train_logs

Training is run by using

```
sbatch horovod.sbatch
```

Code is currently configured to use 4 nodes each with 4 gpus. #allocations changed on midway, this is now 1 node with 4 gpus 
This can be adjusted by changing #SBATCH --nodes=4 #This must now be #SBATCH --gres:gpu=4

It can take a while to schedule 4 nodes. 1 node generally gets scheduled in not
too long.

Code is mainly configured with config.py

Some data augmentation steps are performed in data.py

Current data augmentation includes random rotation around a random center in image, image resizing, left/right flip, up/down
flip, brightness scaling, and gamma scaling. 

AUC over iterations added to TB
