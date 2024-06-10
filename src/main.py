from glob                       import glob
from sklearn.model_selection    import train_test_split

import numpy                    as np
import tensorflow               as tf
import block                    as external_models
import os

from config_loader              import ConfigLoader
from data_preprocessor          import DataPreprocessor
from data_generator             import DataGenerator
from data_normalizer            import DataNormalizer
from model                      import Model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.debugging.set_log_device_placement(True)

config = ConfigLoader('configs')

root_files = glob(str(config.TRAINING_DATA_PATH / '*.root'))
root_files = np.sort(root_files)

train_files, val_files = train_test_split(
    root_files,
    train_size=config.NUM_TRAIN_FILES, 
    test_size=config.NUM_VAL_FILES, 
    shuffle=config.SHUFFLE_FILES
)

normalizer = DataNormalizer(config, val_files, "val")
val_data = DataPreprocessor(config, val_files, "val")
train_data = DataPreprocessor(config, train_files, "train")

graph_net_model = external_models.BlockModel(
    global_output_size=config.OUTPUT_DIMENSIONS, 
    model_config=config.get_model_configs()
)

model = Model(
    config_loader=config, 
    model=graph_net_model,
    val_data=DataGenerator(config, val_files, "val"),
    train_data=DataGenerator(config, train_files, "train"),
)


model.train_model()



