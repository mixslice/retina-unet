###################################################
#
#   Script to execute the prediction
#
##################################################

import os
import ConfigParser

#config file to read from
config = ConfigParser.RawConfigParser()
config.readfp(open(r'./configuration.txt'))
#===========================================
#name of the experiment!!
name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('testing settings', 'nohup')  #std output on log file?

#create a folder for the results if not existing already
result_dir = name_experiment
print "\n1. Create directory for the results (if not already existing)"
if not os.path.exists(result_dir):
    os.system('mkdir -p ' + result_dir)

# finally run the prediction
if nohup:
    print "\n2. Run the prediction on GPU  with nohup"
    os.system('nohup python -u ./src/predict.py > ./{}/{}_prediction.nohup'.
              format(name_experiment, name_experiment))
else:
    print "\n2. Run the prediction on GPU (no nohup)"
    os.system('python ./src/predict.py')
