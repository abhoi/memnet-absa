import os
import pprint
import tensorflow as tf

from nltk import word_tokenize
from data import *
from model import MemN2N
from sklearn.model_selection import StratifiedKFold
import numpy as np

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 300, "internal state dimension [300]")
flags.DEFINE_integer("lindim", 300, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 7, "number of hops [7]")
flags.DEFINE_integer("batch_size", 1, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 10, "number of epoch to use during training [100]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.01, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 40, "clip gradients to this norm [50]")
flags.DEFINE_string("pretrain_file", "data/glove.840B.300d.txt",
                    "pre-trained glove vectors file path [../data/glove.6B.300d.txt]")
flags.DEFINE_string("train_data", "data/Laptops_Train.xml.seg",
                    "train gold data set path [./data/Laptops_Train.xml.seg]")
flags.DEFINE_string("test_data", "data/Laptops_Test_Gold.xml.seg",
                    "test gold data set path [./data/Laptops_Test_Gold.xml.seg]")
flags.DEFINE_boolean("show", False, "print progress [False]")

FLAGS = flags.FLAGS

def read_test_file():
	

def main(_):
	source_count, target_count = [], []
    source_word2idx, target_word2idx, word_set = {}, {}, {}
    max_sent_len = -1

    
if __name__ == '__main__':
    tf.app.run()