# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 22:26:15 2019

@author: Sabarish Sivanath
"""

import numpy as np
import colorlog
import time
import logging
import tensorflow as tf
import configs
from model.model import CSMN
from utils.configuration import ModelConfig
import utils.data_reader as dr
from datetime import datetime
from pathlib import Path

FLAGS = tf.app.flags.FLAGS
TOWER_NAME = 'tower'

def tower_loss(inputs, scope):
    '''
    Runs the data through the model and returns loss.
    Inputs:
        inputs  - tf.data.Dataset iterator. It is dictionary containing input data.
        scope   - scope
    Outputs:
        loss    - model loss
    '''
    net = CSMN(inputs, ModelConfig(FLAGS))
    loss = net.loss
    tf.summary.scalar(scope + 'loss', loss)
    return loss

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
    
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
    
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    colorlog.basicConfig(
        filename=None,
        level=logging.INFO,
        format="%(log_color)s[%(levelname)s:%(asctime)s]%(reset)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.95)
    tf_sess_config = tf.ConfigProto(allow_soft_placement = True,\
                                    log_device_placement = False, \
                                    gpu_options = gpu_options)
    
    with tf.Session(config = tf_sess_config) as sess:
        global_step = tf.get_variable('global_step', [], \
                                      initializer = tf.constant_initializer(0), \
                                      trainable = False)
        #data iterator
        num_elements, data_iter = dr.get_data('train.txt', True)
        
        num_batches_per_epoch = num_elements // FLAGS.BATCH_SIZE // FLAGS.NUM_GPUS
        decay_steps = num_batches_per_epoch * FLAGS.NUM_EPOCHS_PER_DECAY
        
        lr = tf.train.exponential_decay(FLAGS.INIT_LR,
                                        global_step,
                                        decay_steps,
                                        FLAGS.LR_DECAY_FACTOR,
                                        staircase = True)
        opt = tf.train.AdamOptimizer(lr)
        
        tower_gradients = []
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for i in range(FLAGS.NUM_GPUS):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                        data = data_iter.get_next()
                        loss = tower_loss(data, scope)
                        tf.get_variable_scope().reuse_variables()
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        gradients = opt.compute_gradients(loss)
                        tower_gradients.append(gradients)
        gradients = average_gradients(tower_gradients)
        
        summaries.append(tf.summary.scalar('learning_rate', lr))
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], FLAGS.MAX_GRAD_NORM), gv[1]) for gv in gradients]
        
        apply_gradient_op = opt.apply_gradients(clipped_grads_and_vars, global_step = global_step)
        
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 200)
        
        summary_op = tf.summary.merge(summaries)
        
        init = tf.global_variables_initializer()
        sess.run(init)
        
        ckpt = tf.train.get_checkpoint_state(FLAGS.TRAIN_DIR)
        
        if FLAGS.CKPT_RESTORE and ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        summary_writer = tf.summary.FileWriter(FLAGS.TRAIN_DIR, sess.graph)
        
        for step in range(FLAGS.MAX_STEPS):
            start_time = time.time()
            _, loss_value = sess.run([apply_gradient_op, loss])
            duration = time.time() - start_time
            
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            
            if (step + 1)%10 == 0:
                num_data_per_step = FLAGS.BATCH_SIZE * FLAGS.NUM_GPUS
                data_per_sec = num_data_per_step / duration
                sec_per_batch = duration / FLAGS.NUM_GPUS
                
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ''sec/batch)')
                
                c_g_step = int(global_step.eval(session = sess))
                print (format_str % (datetime.now(), c_g_step, loss_value, \
                                     data_per_sec, sec_per_batch))
                
            if (step + 1) % 25 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, c_g_step)
                
            if (step + 1) % 500 == 0 or (step + 1) == FLAGS.MAX_STEPS:
                checkpoint_path = str(Path(FLAGS.TRAIN_DIR, 'model.ckpt'))
                saver.save(sess, checkpoint_path, global_step  = c_g_step)

def main(argv=None):
    if not tf.gfile.Exists(FLAGS.TRAIN_DIR):
        tf.gfile.MakeDirs(FLAGS.TRAIN_DIR)
    train()

if __name__ == "__main__":
    tf.app.run()             