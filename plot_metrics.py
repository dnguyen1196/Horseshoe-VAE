import matplotlib.pyplot as plt

inputstr = """epoch:  0  - objective loss:  250.9177  - train accuracy:  0.5917  - test accuracy:  0.5879
epoch:  10  - objective loss:  82.7879  - train accuracy:  0.5483  - test accuracy:  0.5376
epoch:  20  - objective loss:  61.4148  - train accuracy:  0.5902  - test accuracy:  0.5823
epoch:  30  - objective loss:  54.5258  - train accuracy:  0.6642  - test accuracy:  0.6578
epoch:  40  - objective loss:  51.2453  - train accuracy:  0.6616  - test accuracy:  0.6509
epoch:  50  - objective loss:  62.9572  - train accuracy:  0.6647  - test accuracy:  0.6502
epoch:  60  - objective loss:  52.3146  - train accuracy:  0.6613  - test accuracy:  0.6457
epoch:  70  - objective loss:  51.3994  - train accuracy:  0.6857  - test accuracy:  0.6684
epoch:  80  - objective loss:  51.1796  - train accuracy:  0.6632  - test accuracy:  0.6523
epoch:  90  - objective loss:  50.1201  - train accuracy:  0.6807  - test accuracy:  0.6677
epoch:  100  - objective loss:  59.3688  - train accuracy:  0.6859  - test accuracy:  0.6759
epoch:  110  - objective loss:  48.9149  - train accuracy:  0.6962  - test accuracy:  0.6836
epoch:  120  - objective loss:  47.4816  - train accuracy:  0.6671  - test accuracy:  0.6558
epoch:  130  - objective loss:  48.6656  - train accuracy:  0.6838  - test accuracy:  0.6716
epoch:  140  - objective loss:  47.0295  - train accuracy:  0.6858  - test accuracy:  0.6691
epoch:  150  - objective loss:  48.4605  - train accuracy:  0.6851  - test accuracy:  0.6665
"""

import re


def extract_info_from_input(input):
    metrics_list = list()
    lines = input.split("\n")
    for line in lines:
        metrics = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if not metrics:
            continue
        metrics_list.append(metrics)
    return metrics_list
        
metrics  = extract_info_from_input(inputstr)
# print(metrics)

epochs  =[float(info[0]) for info in metrics]

# NOTE: use one of these for elbo loss vs elbos
# elbos   =[float(info[1]) for info in metrics]
elbos   =[-float(info[1]) for info in metrics]

# NOTE: use one of the following for error vs accuracy


# train   =[info[2] for info in metrics]
# test    =[info[3] for info in metrics]

train   =[1. - float(info[2]) for info in metrics] 
test    =[1. - float(info[3]) for info in metrics]


train_label = 'Link prediction error rate on train data' # NOTE THE LABEL
test_label  = 'Link prediction error rate on test data'


model_name = "L1-VAE [600]" # <<<<< NOTE THE MODEL NAME

plt.plot(epochs, train, '-', color='b', label=train_label)
plt.plot(epochs, test, '--', color='r', label=test_label)
plt.ylim((0, 1))
plt.ylabel('Error rate')
plt.xlabel('Epochs')
plt.title("Error rate vs epoch for {}".format(model_name))
plt.legend()
plt.show()

plt.plot(epochs, elbos, '-', color='r', label='ELBO')
plt.ylabel('ELBO')
plt.xlabel('Epochs')
plt.title("ELBO vs epochs for {}".format(model_name))
plt.show()