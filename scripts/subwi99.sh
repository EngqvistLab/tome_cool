#!/bin/bash
#SBATCH -A C3SE2020-1-14 # group to which you belong
#SBATCH -p vera  # partition (queue)
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -o ../results/ml_classical_models/running_xbox_weighted_inc99.log        # %N for name of 1st allocated node
#SBATCH -t 1-00:00:00                # walltime limit
#SBATCH --mail-user=gangl@chalmers.se
#SBATCH --mail-type=end                 # send mail when job ends
module load GCC/6.4.0-2.28  CUDA/9.1.85  OpenMPI/2.1.2
module load Python/3.6.7
source /c3se/NOBACKUP/users/gangl/Tools/my_python3_vera/bin/activate

#python run_model_evaluation_save_foldindex_weighted.py ../data/dimer_updated_with_madin_bowman_uniform_weights.csv ../results/ml_classical_models/


i=99
python run_model_evaluation_save_foldindex.py \
--infile ../data/dimer_updated_with_madin_bowman_0."$i"_weights.csv \
--weighted \
--outdir ../results/ml_classical_models/ 
