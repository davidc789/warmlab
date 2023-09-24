#!/bin/bash

# Resources
# =========
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --job-name="Warm Simulation Test"
#SBATCH --account="punim2076"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10240
#SBATCH --time=0-1:00:00

# IO and Mailing
# ==============
#SBATCH -o ./logs/slurm.%N.%j-out.log # STDOUT
#SBATCH -e ./logs/slurm.%N.%j-err.log # STDERR
#SBATCH --mail-user=chdc@student.unimelb.edu.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# Modules and Environment Setup
# =============================

# Check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 100
fi

# Run the job from this directory:
cd /data/gpfs/projects/punim2076/warmlab/

module purge
module load GCCcore/11.3.0 Python/3.10.4 SQLite/3.42.0
source ./envs/warmlab/bin/activate

# Actual jobs
# ===========
let NUMPY_THREADS=1
let CORES=${SLURM_CPUS_PER_TASK}

export OMP_NUM_THREADS=${NUMPY_THREADS}
export OPENBLAS_NUM_THREADS=${NUMPY_THREADS}
export MKL_NUM_THREADS=${NUMPY_THREADS}
export VECLIB_MAXIMUM_THREADS=${NUMPY_THREADS}
export NUMEXPR_NUM_THREADS=${NUMPY_THREADS}

hpcsim --cores ${SLURM_CPUS_PER_TASK} --config ./data/jobs/sample_job.json
