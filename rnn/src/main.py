import tensorflow as tf
from tensorflow import keras
import data_helpers
from tensorflow.contrib import learn
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Embedding
from keras.optimizers import Adam
from keras.callbacks import Callback

TF_CPP_MIN_LOG_LEVEL=2

def main():  
    
    # file paths
    file_path_dem = "../data/twtdemtrain.txt"
    file_path_rep = "../data/twtreptrain.txt"
    save_filename = 'savedata.pk'
    save_weights = 'weights.h5'
    
    # hyperparameters
    lr = 0.001 # learning rate
    beta_1 = 0.9 # adam optimizer
    beta_2 = 0.999 # adam optimizer
    valid_split = 0.1 # % of train data to set aside for validation
    num_epochs = 20
    batch_size = 128
    batch_print_int = 10 # number of batches to run before printing loss during training
    
#    # load and preprocess data
#    x_train, y_train, vocab_processor = preprocess(file_path_dem, file_path_rep)
#    
#    # save data to file (so no need to re-load data every run)
#    save_data(save_filename, [x_train, y_train, vocab_processor])

    # open data from file (if saved)
    x_train, y_train, vocab_processor = open_data(save_filename)
    
    # build RNN model
    max_words = len(vocab_processor.vocabulary_)
    max_len = x_train.shape[1]
    model = build_RNN(max_words, max_len)
    
    # train/vaildate RNN
    model, history = train_valid_RNN(model, x_train, y_train, lr, beta_1, beta_2, num_epochs, batch_size, valid_split, batch_print_int)
    
    # save model weights
    model.save_weights(save_weights)
    
#    # load model weights
#    model.load_weights(save_weights)
    
    # predict using model
    
    
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)    

def save_data(save_filename, save_obj):
    print('Saving data...')
    
    with open(save_filename, 'wb') as fi:
        pickle.dump(save_obj, fi)
        
    print('Save data complete. Object saved as:', save_filename, '\n')

def open_data(save_filename):
    print('Opening data...')
    
    with open(save_filename, 'rb') as fi:
        load_temp = pickle.load(fi)
    x_train, y_train, vocab_processor = load_temp[0], load_temp[1], load_temp[2]            

    print('Open data complete.')
    print('Features Shape:', x_train.shape)
    print('Labels Shape:', y_train.shape, '\n')
    
    return x_train, y_train, vocab_processor       
def preprocess(file_path_dem, file_path_rep):
    
    # inputs the positive and negative examples
    
    del_all_flags(tf.flags.FLAGS)
    tf.flags.DEFINE_string("positive_data_file", file_path_dem, "Data source for the positive data.")
    tf.flags.DEFINE_string("negative_data_file", file_path_rep, "Data source for the negative data.")
    FLAGS = tf.flags.FLAGS

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_train = x[shuffle_indices]
    y_train = y[shuffle_indices]

    del x, y
    
    print('Data Loaded.')
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print('Features Shape:', x_train.shape)
    print('Labels Shape:', y_train.shape, '\n')
    
    return x_train, y_train, vocab_processor
        
def build_RNN(max_words, max_len):
    print('Building model...')
    
    tf.reset_default_graph()
    keras.backend.clear_session()

    model = Sequential()
    model.add(Embedding(max_words, 50, input_length = max_len))
    
    model.add(LSTM(64))
    
    model.add(Dense(256, name = 'FC1'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1, name = 'out_layer'))
    model.add(Activation('sigmoid'))

    model.summary()
    
    print ('Model built.', '\n')
    
    return model

def train_valid_RNN(model, x_train, y_train, lr, beta_1, beta_2, num_epochs, batch_size, valid_split, batch_print_int):
    
    print('Training Model...')
    
    model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr, beta_1, beta_2), metrics = ['accuracy'])
    history = model.fit(x_train, y_train, batch_size = batch_size, epochs = num_epochs, validation_split = valid_split, callbacks = [NBatchLogger(batch_print_int)])
    
    print('Model training complete.', '\n')
    
    return model, history

class NBatchLogger(Callback):
    def __init__(self, display):
        self.seen = 0
        self.display = display

    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:
            print (self.seen)
            
if __name__ == '__main__':
    main()