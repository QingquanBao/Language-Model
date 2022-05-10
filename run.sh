#!/bin/bash
#SBATCH --job-name=hostname 
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/slurm/R-%x.%j.out
#python mymain.py -m model=lstm1 model.dropout=0.15,0.25,0.35,0.45 
python mymain.py -m model=lstm 
#python mymain.py -m model=transformer1,transformer2 model.dropout=0.3,0.5
#python mymain.py -m model=transformer1 model.nhead=4,6,8
#python mymain.py -m model=gru1,gru2 # model.nlayers=3,4 model.dropout=0.25,0.35,0.45 model.clip=0.25,0.5 
#python mymain.py -m model=gru model.clip=0.4,0.8 model.lr=0.1,1,10
#python mymain.py -m model=lstm,lstm1,lstm2,lstm3 
#python mymain.py -m model=lstm model.batch_size=10,20,30,40 model.bptt=20,35,50
