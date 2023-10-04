#!/bin/bash
scp dist/warm-1.1.4.tar.gz chdc@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2076/warmlab
scp scripts/warmlab.slurm chdc@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2076/warmlab/scripts
scp data/jobs/*.json chdc@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2076/warmlab/data/jobs/
scp data/SimDataSummary.csv chdc@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2076/warmlab/data/remoteSimDataSummary.csv
scp data/backup/*.csv chdc@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2076/warmlab

scp chdc@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2076/warmlab/data/SimData.csv data/remoteSimData.csv
scp chdc@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2076/warmlab/data/SimInfo.csv data/remoteSimInfo.csv
scp chdc@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2076/warmlab/data/SimDataSummary.csv data/remoteSimDataSummary.csv
