#!/bin/bash

#BSUB -n 16
#BSUB -R "span[ptile=16]"
#BSUB -R gpu
#BSUB -q "standard"
#BSUB -o sing_tf.out
#BSUB -e sing_tf.err
#BSUB -J sing_tf

#---------------------------------------------------------------------
module load singularity
cd /extra/vikasy
singularity run --nv tf_gpu-1.2.0-cp35-cuda8-cudnn51.img /extra/vikasy/4chars_Prefix_Suffix_Experiments/Spanish/main.py
