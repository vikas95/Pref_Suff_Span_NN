import os
from general_utils import get_logger


class Config():
    def __init__(self):
        # directory for training outputs
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # create instance of logger
        self.logger = get_logger(self.log_path)
        

    # general config
    #output_path = "/extra/vikasy/4chars_Prefix_Suffix_Experiments/Spanish/results/crf/"
    output_path = "results/crf/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"

    # embeddings
    dim = 300
    dim_char = 50

    dim_pref = 30
    dim_suff = 30

    dim_pref_2 = 20     ## Check the vocab size before assigning this value
    dim_suff_2 = 20

    dim_pref_4 = 30
    dim_suff_4 = 30

    # path1="/extra/vikasy/4chars_Prefix_Suffix_Experiments/Spanish/data/"
    path1 = "data/"
    glove_filename = os.path.join(path1,"wiki.es.vec")
    
    # trimmed embeddings (created from glove_filename with build_data.py)
    trimmed_filename = os.path.join(path1,"wiki.es.vec.trimmed.npz")

    # dataset
    dev_filename = os.path.join(path1,"esp.testa.txt")
    test_filename = os.path.join(path1,"esp.testb.txt")
    train_filename = os.path.join(path1,"esp.train.txt")
    max_iter = None # if not None, max number of examples

    # vocab (created from dataset with build_data.py)
    words_filename = os.path.join(path1,"words.txt")
    tags_filename = os.path.join(path1,"tags.txt")
    chars_filename = os.path.join(path1,"chars.txt")
    PS_filename = os.path.join(path1,"PS.txt")
    PS_filename_2 = os.path.join(path1,"PS_2.txt")
    PS_filename_4 = os.path.join(path1,"PS_4.txt")
    # training
    train_embeddings = True
    nepochs = 150
    dropout = 0.50
    batch_size = 100
    lr_method = "sgd"   # "adam"
    lr = 0.10
    lr_decay = 0.99
    clip = 5 # if negative, no clipping
    nepoch_no_imprv = 30
    reload = False
    
    # model hyperparameters
    hidden_size = 50   #300
    char_hidden_size = 25   #100
    pref_hidden_size=15
    suff_hidden_size=15

    pref_hidden_size_2 = 10
    suff_hidden_size_2 = 10

    pref_hidden_size_4 = 15
    suff_hidden_size_4 = 15

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    crf = True # if crf, training is 1.7x slower on CPU
    chars = True # if char embedding, training is 3.5x slower on CPU
    pref_suff = True

