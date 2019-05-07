import pprint
import os
import json
from pathlib import Path
import colorlog

pp = pprint.PrettyPrinter().pprint


class BaseConfig(object):
    def __init__(self, FLAGS):
        attrs = FLAGS.flag_values_dict()
        for attr in attrs:
            setattr(self, attr.lower(), getattr(FLAGS, attr))

    def save(self, name):
        fname = os.path.join(self.train_dir, name)
        if not os.path.exists(fname):
            with open(fname, 'w') as f:
                json.dump(vars(self), f, indent = 4, sort_keys = True)

    def load(self, name):
        fname = os.path.join(self.train_dir, name)
        if os.path.exists(fname):
            with open(fname, 'r') as f:
                contents = json.load(f)
            for key, value in contents.items():
                setattr(self, key, value)


class ModelConfig(BaseConfig):
    def __init__(self, FLAGS):
        super(ModelConfig, self).__init__(FLAGS)

        # Embedding dimensions
        self.img_dim = 2048

        # Memory size
        self.img_memory_size = 1
        if self.use_user_context:
            self.memory_size = self.img_memory_size + self.max_context_length + self.max_output_length
        else:
            self.memory_size = self.img_memory_size + self.max_output_length

        # Memory CNN
        self.context_filter_sizes = [3, 4, 5]
        self.output_filter_sizes = [3, 4, 5]
        if self.use_user_context:
            self.num_channels_total = self.num_channels * (len(self.context_filter_sizes) + len(self.output_filter_sizes))
        else:
            self.num_channels_total = self.num_channels * len(self.output_filter_sizes)
                             
        colorlog.info("Model configuration")
        pp(vars(self))
        self.batch_size = FLAGS.BATCH_SIZE
        self.save("model.config")


class TrainingConfig(BaseConfig):
    def __init__(self, FLAGS):
        super(TrainingConfig, self).__init__(FLAGS)

        colorlog.info("Training configuration")
        pp(vars(self))
        self.load("training.config")
        self.save("training.config")
