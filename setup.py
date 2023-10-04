#!/usr/bin/env python

import os
from setuptools import setup, find_packages

# Version.Subversion.BuildNumber
version = "1.1.4"
requirement_path = "warmlab/local_worker/requirements.txt"

install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setup(
    name="warm",
    version=version,
    description="Warm core module",
    author="David Chen",
    author_email="david.chen@student.unimelb.edu.au",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        "hpc": ["tqdm"],
        "analyser": ["plotly", "matplotlib", "dash", "dash-cytoscape"],
        "manager": ["aiohttp", "flask", "waitress"]
    },
    entry_points={
        "console_scripts": [
            "hpcsim = warmlab.hpc.main:main [hpc]",
            "warmlab = warmlab.manager.main:main [manager]"
        ],
    }
)
