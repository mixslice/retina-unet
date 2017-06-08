###################################################
#
#   Script to launch the training
#
##################################################

import os
import ConfigParser

#config file to read from
config = ConfigParser.RawConfigParser()
config.readfp(open(r'./configuration.txt'))
#===========================================
#name of the experiment
name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('training settings',
                          'nohup')  #std output on log file?

#create a folder for the results
result_dir = name_experiment
print "\n1. Create directory for the results (if not already existing)"
if os.path.exists(result_dir):
    print "Dir already existing"
else:
    os.system('mkdir -p ' + result_dir)

print "copy the configuration file in the results folder"
os.system('cp configuration.txt ./{}/{}_configuration.txt'.format(
    name_experiment, name_experiment))

# run the experiment
if nohup:
    print "\n2. Run the training on GPU with nohup"
    os.system('nohup python -u ./src/train.py > ./{}/{}_training.nohup'.
              format(name_experiment, name_experiment))
else:
    print "\n2. Run the training on GPU (no nohup)"
    os.system('python ./src/train.py')

#Prediction/testing is run with a different script
