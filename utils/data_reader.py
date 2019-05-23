from pathlib import Path
import tensorflow as tf
import numpy as np
from scripts.generate_dataset import GO_ID, EOS_ID
from random import shuffle

FLAGS = tf.flags.FLAGS

def numpy_read_func(np_file_path):
    '''
    Reads numpy file.
    
    args:
        np_file_path - numpy file name
        
    returns:
        numpy array
    '''
    return np.load(str(Path(FLAGS.img_data_dir, np_file_path.decode("utf-8"))))


def token_split_func(token, max_length, cap=None):
    id_ = np.zeros([ max_length], dtype=np.int32)

    if b'_' in token:
        valid_ids = list(map(int, token.decode("utf-8").split('_')))[:max_length]    
    elif len(token) == 0:
        valid_ids = []
    else:
        valid_ids = [int(token)]
        
    if cap:
        valid_ids = valid_ids[:max_length-1]
        if cap == 'caption':
            id_[len(valid_ids) + 1] = [GO_ID] + valid_ids[:]
        else:
            id_[:len(valid_ids) + 1] = valid_ids[:] + [EOS_ID]
    else:
        id_[:len(valid_ids)] = valid_ids[:]
    return id_

def mask_build_func(length, max_length, cap = None):
    mask = np.zeros([max_length], dtype = np.bool)
    if cap:
        length = int(length) + 1
    else:
        length = int(length)
    
    length = max_length if length > max_length else length
    mask[:length] = True
    return mask

def _parse_string(filename):
    lsplit = tf.string_split([filename], delimiter = ',').values
    context_len = tf.minimum(tf.cast(tf.string_to_number(lsplit[1]), tf.int32), FLAGS.max_context_length)
    caption_len = tf.minimum(tf.cast(tf.string_to_number(lsplit[2]), tf.int32) + 1, FLAGS.max_output_length)
    img_embedding = tf.py_func(numpy_read_func, [lsplit[0]], tf.float32)
    context_id = tf.py_func(token_split_func, [lsplit[3], FLAGS.max_context_length], tf.int32)
    caption_id = tf.py_func(token_split_func, [lsplit[4], FLAGS.max_output_length, 'caption'], tf.int32)
    answer_id = tf.py_func(token_split_func, [lsplit[4], FLAGS.max_output_length, 'answer'], tf.int32)
    context_mask = tf.py_func(mask_build_func, [context_len, FLAGS.max_context_length], tf.bool)
    caption_mask = tf.py_func(mask_build_func, [caption_len, FLAGS.max_output_length, 'caption'], tf.bool)
    
    
    img_embedding.set_shape([2048, 1, 1])
    img_embedding = tf.transpose(tf.reshape(img_embedding, [2048, 1]), perm = [1,0])
    context_id.set_shape(FLAGS.max_context_length)
    caption_id.set_shape(FLAGS.max_output_length)
    answer_id.set_shape(FLAGS.max_output_length)
    context_mask.set_shape(FLAGS.max_context_length)
    caption_mask.set_shape(FLAGS.max_output_length)
    

    #return tf.concat([lsplit[0], context_len, caption_len, lsplit[3], lsplit[4]], 0)
    return {"img_embed" : img_embedding, 
            "context_len" : context_len,
            "caption_len" : caption_len,
            "context_id" : context_id,
            "caption_id" : caption_id,
            "answer_id" : answer_id,
            "context_mask" : context_mask,
            "caption_mask" : caption_mask}
    

def get_data(filepath, is_train = True):
    filenames = []

    filenames = [l.strip() for l in open(str(Path(FLAGS.data_dir, filepath))).readlines()]
    if is_train:
        shuffle(filenames)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(lambda x: _parse_string(x))
    if is_train:
        dataset = dataset.shuffle(buffer_size = 40*FLAGS.BATCH_SIZE)
    dataset = dataset.batch(FLAGS.BATCH_SIZE, drop_remainder = True).prefetch(FLAGS.PREFETCH_MULT*FLAGS.BATCH_SIZE)
    if is_train:
        dataset = dataset.repeat(-1)
    
    iterator = dataset.make_one_shot_iterator()

    return len(filenames), iterator