# 

## Deployment Guide

0. Copy all the files over to a local directory.
1. Install Python 3.11 and make sure it is on `PATH`.
2. In console, change to the project folder and run
```shell
pip install -r requirements.txt
```
If prompted, upgrade `pip` just to be safe:
```shell
python -m pip install --upgrade pip
```
3. Install all dependencies with 
```shell
pip install -r requirements.txt
```
4. Check your LAN IP address. Start the server with
```shell
waitress-serve [LAN IP] app:app
```
5. You are good to go!
