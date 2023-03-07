import argparse
import numpy as np
import torch
import pandas as pd
from pprint import pprint
import random

from data import Data
from utils import get_dataset, get_net, get_strategy
from config import parse_args
from seed import setup_seed
from visualization import visualiazation

args = parse_args()
pprint(vars(args))
print()

# fix random seed
setup_seed(42)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# load dataset
X_train, Y_train, X_val, Y_val, X_test, Y_test, handler, full_test_imgs_list, x_test_slice, test_brain_images, test_brain_masks = get_dataset(args.dataset_name,supervised=False)
dataset = Data(X_train, Y_train, X_val, Y_val, X_test, Y_test, handler)

net = get_net(args.dataset_name, device) # load network
strategy = get_strategy(args.strategy_name)(dataset, net) # load strategy
target_num = dataset.cal_target()

# start experiment
dataset.initialize_labels_random(args.n_init_labeled)
print("Round 0")
rd = 0
strategy.train(rd, args.training_name)
accuracy = []
size = []
preds= strategy.predict(dataset.get_test_data(), full_test_imgs_list, x_test_slice, test_brain_images) # get model prediction for test dataset
print(f"Round 0 testing accuracy: {dataset.cal_test_acc(preds,test_brain_masks)}")  # get model performance for test dataset
accuracy.append(dataset.cal_test_acc(preds,test_brain_masks))
size.append(args.n_init_labeled)
testing_accuracy = 0

# pseudo label filter
unlabeled_idxs, unlabeled_data = dataset.get_unlabeled_data(index = None)
labels = net.predict_black_patch(unlabeled_data)
index = dataset.delete_black_patch(unlabeled_idxs, labels)

unlabeled_idxs, unlabeled_data = dataset.get_unlabeled_data(index = index)
print(f"number of labeled pool: {args.n_init_labeled}")
print(f"number of unlabeled pool: {len(unlabeled_idxs)}")
print(f"number of testing pool: {dataset.n_test}")
print()

# active learning process
query_samples = []
adversarial_samples = []
for rd in range(1, args.n_round + 1):
    print(f"Round {rd}")
    # query
    if args.strategy_name == "AdversarialAttack":
        query_idxs, generative_top_sample = strategy.query(args.n_query,index) #([500, 1, 128, 128])
        label = dataset.get_label(query_idxs)
        # generated adversarial sample expansion
        X_train_expansion = dataset.add_labeled_data(generative_top_sample, label)
        query_samples.append(query_idxs)
        print(query_idxs)
        if rd==1:
            visualiazation(X_train_expansion, query_samples, target_num, rd, args.strategy_name)
            break
    else:
        query_idxs = strategy.query(args.n_query,index)  # query_idxs为active learning请求标签的数据

    # update labels
    strategy.update(query_idxs)  # update training dataset and unlabeled dataset for active learning
    strategy.train(rd, args.training_name)

    # calculate accuracy
    preds= strategy.predict(dataset.get_test_data(), full_test_imgs_list, x_test_slice, test_brain_images)
    testing_accuracy = dataset.cal_test_acc(preds,test_brain_masks)
    print(f"Round {rd} testing accuracy: {dataset.cal_test_acc(preds,test_brain_masks)}")

    accuracy.append(testing_accuracy)
    labeled_idxs, _ = dataset.get_labeled_data()
    size.append(len(labeled_idxs))

# save the result
dataframe = pd.DataFrame(
    {'model': 'Unet', 'Method': args.strategy_name, 'Training dataset size': size, 'Accuracy': accuracy})
dataframe.to_csv(f"./{args.strategy_name}.csv", index=False, sep=',')