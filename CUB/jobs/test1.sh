#!/bin/bash
#SBATCH --mail-user=ar.aamer@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=cbm_test
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=0-00:20
#SBATCH --account=rrg-ebrahimi

nvidia-smi

source ~/my_env/cbm/bin/activate

echo "------------------------------------< Data preparation>----------------------------------"
echo "Copying the source code"
date +"%T"
cd $SLURM_TMPDIR
cp -r ~/scratch/ConceptBottleneck .


echo "Copying the datasets"
date +"%T"
cp -r ~/scratch/Datasets/CUB_200_2011.tar.gz .
cp -r ~/scratch/Datasets/places365.tar.gz .
cp -r ~/scratch/Datasets/pretrained.tar.gz .
cp -r class_attr_data_10.zip .
tar -xvzf places365.tar.gz
tar -xvzf CUB_200_2011.tar.gz
tar -xvzf pretrained.tar.gz
unzip class_attr_data_10.zip


mkdir $SLURM_TMPDIR/ConceptBottleneck/CUB/logs/test1

cd ..
echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"
cd $SLURM_TMPDIR

cd ConceptBottleneck

python3 experiments.py cub Concept_XtoC --seed 1 -ckpt 1 -log_dir logs/test1 -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir $SLURM_TMPDIR/class_attr_data_10 -n_attributes 112 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -bottleneck 

echo "-----------------------------------<End of run the program>---------------------------------"
date +"%T"
echo "--------------------------------------<backup the result>-----------------------------------"
date +"%T"
cd $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/ConceptBottleneck/CUB/logs/test1 ~/scratch/ConceptBottleneck/CUB/logs