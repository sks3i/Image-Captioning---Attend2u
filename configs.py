import tensorflow as tf

flags = tf.app.flags


#Data input & storage
flags.DEFINE_string("data_dir", "./data/caption_dataset", 
                    "Caption dataset path" )

flags.DEFINE_string("img_data_dir", "./data/resnet_pool5_features", 
                    "Image feaature files path")

flags.DEFINE_integer('PREFETCH_MULT', 2,
                     'Data to prefetch: prefect_mult*batch_size')

flags.DEFINE_string('TRAIN_DIR', './checkpoints', 
                    'Directory where to write event logs and checkpoint.')

#Model hyperparams
flags.DEFINE_integer("max_context_length", 60, 
                     "User contex max length default [60]" )

flags.DEFINE_integer("max_output_length", 16, 
                     "Max output length.")

flags.DEFINE_integer('BATCH_SIZE', 4, 
                     "Number of examples to process in a batch." )

flags.DEFINE_integer('vocab_size', 40000, 
                     "Number of vocab.")

flags.DEFINE_integer('word_emb_dim', 512,
                     "Dimensions of word embeddings.")

flags.DEFINE_integer('mem_dim', 1024, 
                     "Dimensions of memories.")

flags.DEFINE_integer('num_channels', 300, 
                     "Number of channels of memory cnn.")

flags.DEFINE_bool('CKPT_RESTORE', False,
                  'To restore model from checkpoint')

flags.DEFINE_bool('USE_USER_CONTEXT', False,
                  "Enable/Disable user context")

flags.DEFINE_float("INIT_LR", 0.001, 
                   "initial learning rate [0.01]")

flags.DEFINE_float("MAX_GRAD_NORM", 100, 
                   "clip gradients to this norm [100]")

flags.DEFINE_integer("MAX_STEPS", 500000, 
                     "number of steps to use during training [500000]")

flags.DEFINE_integer("NUM_EPOCHS_PER_DECAY", 8, 
                     "Epochs after which learning rate decays")

flags.DEFINE_float("MOV_AVG_DECAY", 0.9999, 
                   "Decay to use for moving average")

flags.DEFINE_float("LR_DECAY_FACTOR", 0.8, 
                   "Learning rate decay factor")

flags.DEFINE_integer("NUM_GPUS", 1, 
                     "Number of gpus to use")