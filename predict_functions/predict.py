import os
import pprint
import tensorflow as tf

from nltk import word_tokenize
from predict_data import *
from predict_model import MemN2N
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 300, "internal state dimension [300]")
flags.DEFINE_integer("lindim", 300, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 7, "number of hops [7]")
flags.DEFINE_integer("batch_size", 1, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 100, "number of epoch to use during training [100]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 50, "clip gradients to this norm [50]")
flags.DEFINE_string("test_data", "tech", "test data [tech, food]")
flags.DEFINE_string("pretrain_file", "../data/glove.840B.300d.txt", "pre-trained glove vectors file path [../data/glove.6B.300d.txt]")
# flags.DEFINE_string("train_data", "data/Laptops_Train.xml.seg", "train gold data set path [./data/Laptops_Train.xml.seg]")
# flags.DEFINE_string("test_data", "data/Laptops_Test_Gold.xml.seg", "test gold data set path [./data/Laptops_Test_Gold.xml.seg]")
flags.DEFINE_boolean("show", False, "print progress [False]")

FLAGS = flags.FLAGS

def predict_labels(model, test_data):
    source_data, source_loc_data, target_data = test_data
    print(model.get_operations())


def main(_):
    source_count, target_count = [], []
    source_word2idx, target_word2idx, word_set = {}, {}, {}
    max_sent_len = -1

    max_sent_len = get_dataset_resources_test(FLAGS.test_data, source_word2idx, target_word2idx, word_set, max_sent_len)
    max_sent_len = get_dataset_resources_test(FLAGS.test_data, source_word2idx, target_word2idx, word_set, max_sent_len)
    embeddings = load_embedding_file(FLAGS.pretrain_file, word_set)

    print("max sentence length: " + str(max_sent_len))
    FLAGS.pad_idx = source_word2idx['<pad>']
    FLAGS.nwords = len(source_word2idx)
    FLAGS.mem_size = max_sent_len

    pp.pprint(flags.FLAGS.__flags)

    # print('loading pre-trained word vectors...')
    # print('loading pre-trained word vectors for train and test data')

    FLAGS.pre_trained_context_wt, FLAGS.pre_trained_target_wt = get_embedding_matrix(embeddings, source_word2idx, target_word2idx, FLAGS.edim)

    test_data = get_dataset_test(FLAGS.test_data, source_word2idx, target_word2idx, embeddings)
    # source_data, source_loc_data, target_data = test_data
    # X = np.column_stack((source_data, source_loc_data, target_data))

    # y = np.array(target_label)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # train_data_inner = (X_train[:,0], X_train[:,1], X_train[:,2], y_train)
    # test_data_inner = (X_test[:,0], X_test[:,1], X_test[:,2], y_test)

    with tf.Session() as sess:
        print("accessing model...")
        model = MemN2N(FLAGS, sess)
        model.build_model()
        saver = tf.train.import_meta_graph('../models/memnet-' + FLAGS.test_data + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint('../'))
        model.run(test_data)
        # model.run(test_data)
        #model.build_model()
        #model.run(test_data)
        # model = tf.get_default_graph()
        # predict_labels(model, test_data)
        print("model restored!")
        # model = MemN2N(FLAGS, sess)
        # model.build_model()
        # saver = tf.train.Saver()
        # model.run(train_data_inner, test_data_inner)
        # saver.save(sess, './memnet-food')

    # use this to restore model from disk
    # sess = tf.Session()
    # saver = tf.train.import_meta_graph('./memnet-1000.meta')
    # saver.restore(sess, tf.train.latest_checkpoint('./'))

if __name__ == '__main__':
    tf.app.run()
