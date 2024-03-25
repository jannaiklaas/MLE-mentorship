import numpy as np
import random
import tensorflow as tf
import os

def set_seed(seed):
    """Set the seed for reproducibility."""
    tf.random.set_seed(42)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)