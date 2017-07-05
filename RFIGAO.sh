#! /bin/bash
#SBATCH -t 0:30:00
#SBATCH -n 10                                                         
#SBATCH --mem=10G                                                               
#SBATCH -J HERAGenAlgTest
###SBATCH -p jpober-test
###SBATCH --output=/users/jkerriga/RFIGenAlg/GenAlg_%A_%a.out
####SBATCH --array=0-100:1

source activate PAPER
cd ~/RFIGenAlg/

python runRFIGenAl.py