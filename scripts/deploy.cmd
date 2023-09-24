#!/bin/bash
scp dist/warm-1.1.1.tar.gz chdc@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2076/warmlab
scp scripts/warmlab.slurm chdc@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2076/warmlab/scripts
scp data/jobs/*.json chdc@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2076/warmlab/data/jobs/
scp data/SimDataSummary.csv chdc@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2076/warmlab/data/remoteSimDataSummary.csv

scp chdc@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2076/warmlab/data/batch1/*.csv ./data
scp chdc@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2076/warmlab/data/batch2/*.csv ./data
scp chdc@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2076/warmlab/data/SimDataSummary.csv data/remoteSimDataSummary.csv
