# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_de_vocab, load_en_vocab
from train import Graph
#from nltk.translate.bleu_score import corpus_bleu

def eval(): 
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    # X, Sources, Targets = load_test_data()

    """
        x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [de2idx.get(word, 1) for word in (source_sent + u" </S>").split()] # 1: OOV, </S>: End of Text
        y = [en2idx.get(word, 1) for word in (target_sent + u" </S>").split()] 
        if max(len(x), len(y)) <=hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)
    
    # Pad      
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
    """
    en2idx, idx2en = load_en_vocab()
     
    # Start session         
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))

            while(True):
                prompt = raw_input()
                xlist = []
                xval = [en2idx.get(word, 1) for word in (target_sent + u" </S>").split()]
                if (len(xval) <= hp.maxlen):
                    xlist.append(np.array(xval))
                X = np.zeros([len(xlist), hp.maxlen], np.int32)
                for i, xi in enumerate(xlist):
                    X[i] = np.lib.pad(x, [0, hp.maxlen - len(x)], 'constant', constant_values=(0, 0))
                list_of_refs, hypotheses = [], []
                for i in range(len(X) // hp.batch_size):

                    ### Get mini-batches
                    x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                    prompt = raw_input()
                    ### Autoregressive inference
                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        #print("j: " + str(j))
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]
                    
                    #print(pred) # pred should be length 1 each time due to the cycling of the while loop in main
                    for pred in preds:
                        got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                        #return got
                    print(got)
                                          
if __name__ == '__main__':
    eval()
    