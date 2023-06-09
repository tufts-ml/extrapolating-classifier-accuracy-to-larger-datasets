import os

def makedir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)