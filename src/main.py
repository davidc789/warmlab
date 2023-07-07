from subprocess import Popen
import graphAdapter

proc = Popen(["./simulator/bin/simulator"])
proc.communicate(simulatorInput, 1000)
