import os

def makedir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def one_hot_encode_list(findings, labels):
    encoding = [1 if word in findings else 0 for word in labels]
    return encoding