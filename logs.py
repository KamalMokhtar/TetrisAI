from keras.callbacks import TensorBoard
import tensorflow as tf
from tensorflow.summary import FileWriter
#from tensorboardX import FileWriter

class CustomTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = FileWriter(self.log_dir) #FileWriter(self.log_dir) #tf.summary.create_file_writer(self.log_dir)#

    def set_model(self, model):
        pass

    def log(self, step, **stats):
        self._write_logs(stats, step)
