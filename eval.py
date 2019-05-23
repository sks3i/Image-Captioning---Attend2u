from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import colorlog
import time
import utils.data_reader as dr
from model.model import CSMN
from scripts.generate_dataset import EOS_ID
from utils.configuration import ModelConfig
from utils.evaluator import Evaluator
from termcolor import colored
import configs

FLAGS = tf.app.flags.FLAGS

TOWER_NAME = 'tower'

def load_vocabulary(vocab_fname):
    with open(vocab_fname,  'r') as f:
        vocab = f.readlines()
        
    vocab = [s.strip() for s in vocab]
    
    rev_vocab = {}
    for i, token in enumerate(vocab):
        rev_vocab[i] = token
    
    return vocab, rev_vocab

def inject_summary(key_value):
    summary = tf.Summary()
    for key, value in key_value.items():
        summary.value.add(tag = '%s' % (key), simple_value = value)
    return summary

def eval_once(saver, summary_writer, argmaxs, ans_ids, vocab, rev_vocab, num_data, b_global_step):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.80)
    sess_config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False, gpu_options = gpu_options)
    
    with tf.Session(config = sess_config) as sess:
        print('Hello')
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            
            global_step =  ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            if global_step == b_global_step:
                return global_step
        else:
            print('No checkpoint file found.')
            return
        
        step = 0
        max_iter = num_data // FLAGS.BATCH_SIZE
        
        res = []
        ans = []
        res_tokens_list = []
        ans_tokens_list = []
        while(step < max_iter):
            results = sess.run([argmaxs, ans_ids])
            res += results[0].tolist()
            ans += results[1].tolist()
            step += 1
        
        for i in range(len(res)):
            caption = []
            answer = []
            for k in range(len(res[i])):
                token_id = res[i][k]
                if token_id == EOS_ID:
                    break
                caption.append(rev_vocab[token_id])
            for k in range(len(ans[i])):
                token_id = ans[i][k]
                if token_id == EOS_ID:
                    break
                answer.append(rev_vocab[token_id])
            res_tokens_list.append(caption)
            ans_tokens_list.append(answer)
            
            
        colorlog.info(colored("Validation output example: (%s)" % global_step, 'green'))
        for i, (res, ans) in enumerate(zip(res_tokens_list[:5], ans_tokens_list[:5])):
            print('%d' % i)
            print(' '.join(ans))
            print(' '.join(res))
        
        evaluator = Evaluator()
        result = evaluator.evaluation(res_tokens_list, ans_tokens_list, 'coco')
        
        summary = inject_summary(result)
        summary_writer.add_summary(summary, global_step)
        
        return global_step
            


def evaluate():
    vocab, rev_vocab = load_vocabulary(FLAGS.vocab_fname)
    
    with tf.Graph().as_default() as g:
        n, data_iter =  dr.get_data('test2.txt', False)
        
        tower_argmax = []
        tower_ans_id = []
        
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for i in range(FLAGS.NUM_GPUS):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' %(TOWER_NAME, i)) as var_scope:
                        inp = data_iter.get_next()
                        net = CSMN(inp, ModelConfig(FLAGS), is_training = False)
                        argmax = net.argmax
                        tf.get_variable_scope().reuse_variables()
                        
                        tower_argmax.append(argmax)
                        tower_ans_id.append(inp['answer_id'])
                        
        argmaxs = tf.concat(tower_argmax, 0)
        answer_ids = tf.concat(tower_ans_id, 0)
        saver = tf.train.Saver(tf.global_variables())
        
        summary_op = tf.summary.merge_all()
        
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
        
        b_g_s = "0"
        
        while True:
            c_g_s = eval_once(saver, summary_writer, argmaxs, answer_ids, vocab, 
                              rev_vocab, n, b_g_s)
            b_g_s = c_g_s
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)
    
def main(argv = None):
    if not tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()

if __name__ == '__main__':
    tf.app.run()