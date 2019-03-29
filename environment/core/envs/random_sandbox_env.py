from enum import Enum
import time

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import constants.environment as env_const

# TODO add performance metrics!
class RandomSandboxEnv():
    def __init__(
        self,
        buffer,
        config
        ):
        pass
    
    def randomize(self):
        pass

    
    