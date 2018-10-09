import numpy             as np
import tensorflow        as tf
import pandas            as pd
import matplotlib.pyplot as plt
import pickle
import time
import tqdm
from glob import glob
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.contrib.layers import l2_regularizer, xavier_initializer_conv2d

#lines 15 - 23 was code that was provided by Silverpond during the workshop and not code that I wrote 


!test -e ./data/cifar-10-batches-py || \
    ( wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz \
    && mkdir -p data \
    && tar -C data -xvzf cifar-10-python.tar.gz) 


#conversion into standard format for images
  
data_dir = "./data/cifar-10-batches-py"



#code above provided by Silverpond during workshop. Code below is mine.



# Training data
all_train_batches = glob(f"{data_dir}/data_batch*")
loaded_train = [ pickle.load(open(x, "rb"), encoding="latin1")
                for x in all_train_batches ]
train_images = np.concatenate([x["data"]
                 .reshape(-1, 3, 32, 32)
                 .transpose(0, 2, 3, 1) for x in loaded_train])
train_labels = np.concatenate([x["labels"] for x in loaded_train])

# Test data
test_data   = pickle.load(open(f"{data_dir}/test_batch", "rb"), encoding="latin1")
test_images = test_data["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
test_labels = np.array(test_data["labels"])

# Get the metadata
label_lookup = pickle.load(open(f"{data_dir}/batches.meta", "rb"))["label_names"]




#selection of batches

def sample_batch(images, labels, batch_size=8):
    chosen_indices = np.random.choice(len(images), size=batch_size)
    return images[chosen_indices], labels[chosen_indices]
    
    mean_image = np.mean(train_images, axis=0)
    
