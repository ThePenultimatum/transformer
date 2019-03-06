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

def eval(prompt): 
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    # X, Sources, Targets = load_test_data()
    # print(Sources)
    # print(Targets)
    # print(X)
    # de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
     
#     X, Sources, Targets = X[:33], Sources[:33], Targets[:33]
     
    # Start session         
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                for i in range(len(X) // hp.batch_size):

                    ### Get mini-batches
                    x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                    sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                    targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]
                     
                    ### Autoregressive inference
                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        #print("j: " + str(j))
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]
                    
                    #print(pred) # pred should be length 1 each time due to the cycling of the while loop in main
                    for pred in preds:
                        got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                        return got
                                          
if __name__ == '__main__':
    while(1):
        prompt = raw_input()
        res = eval(prompt)
        print(res)
    
    