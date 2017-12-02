import numpy as np
import os
import unicodedata

max_words_in_sent=0
# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"
label2ind={}
# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


class CoNLLDataset(object):
    label2ind = {}
    """
    Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags
    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```
    """
    def __init__(self, filename, processing_word=None, processing_tag=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield
        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None


    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART12345-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    word, tag = ls[0],ls[-1]
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]


    def __len__(self):
        """
        Iterates once over the corpus to set and store length
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_vocabs(datasets):
    """
    Args:
        datasets: a list of dataset objects
    Return:
        a set of all the words in the dataset
    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocab(dataset):
    """
    Args:
        dataset: a iterator yielding tuples (sentence, tags)
    Returns:
        a set of all the characters in the dataset
    """
    vocab_char = set()
    for words, _ in dataset:
        for word in words:
            vocab_char.update(word)
    # print (vocab_char)
    return vocab_char

#############################################################
#############################################################
# Make suffix-prefix vocab development function here.

def get_Pref_Suff_vocab(dataset):
    """
    Args:
        dataset: a iterator yielding tuples (sentence, tags)
    Returns:
        a set of all the characters in the dataset
    """
    vocab_Pref_Suff = []
    for words1, _ in dataset:
        for words in words1:
            pref=""
            suff=""
            words = str(words).lower()
            # print (words)
            if len(words)>2:

               pref=words[:3]
               suff=words[-3:]
            elif len(words)==2:

                pref = "_"+str(words[:2])
                suff = str(words[:2])+"_"
                #print (pref)
            elif len(words)==1:

                pref = "_" + "_" + str(words[:2])
                suff = str(words[:2]) + "_" + "_"

            vocab_Pref_Suff.append(pref)
            vocab_Pref_Suff.append(suff)
    vocab_Pref_Suff=set(vocab_Pref_Suff)
    print (vocab_Pref_Suff)
    return vocab_Pref_Suff


#############################################################
#############################################################

# Make CASE FEATURE vocab development function here.

def get_case_vocab(dataset):
    """
        Args:
        dataset: a iterator yielding tuples (sentence, tags)
        Returns:
        a set of all the characters in the dataset
        """
    vocab_case = []
    for words1, _ in dataset:
        for words in words1:
            case=""
            words = str(words)
            # print (words)
            chars1=list(words)
            case_chars1="".join([unicodedata.category(c) for c in chars1])
                
            vocab_case.append(case_chars1)

    vocab_case=set(vocab_case)
    print (vocab_case)
    return vocab_case

#############################################################
#############################################################
#############################################################
def get_glove_vocab(filename):
    """
    Args:
        filename: path to the glove vectors
    """
    print("Building vocab...")
    vocab = set()
    with open(filename,encoding="utf-8") as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    """
    Writes a vocab to a file

    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    Returns:
        write a word per line
    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise MyIOError(filename)
    return d

###################################################

### Try to write a function for character embedding also.

###################################################

def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """
    Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)


def get_processing_word(vocab_words=None, vocab_chars=None,vocab_pref_suff=None,vocab_case=None,
                    lowercase=False, chars=False, Pref_Suff=False):
    """
    Args:
        vocab: dict[word] = idx
    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)
    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            #print (word)
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:                   ######################### develop the vocab for suffix prefix and add it in similar way.
                    char_ids += [vocab_chars[char]]

        #if vocab_pref_suff is not None and Pref_Suff== True:
            Pref_ID=0
            Suff_ID=0
            word=str(word).lower()

            if len(word)>2:
               if word[0:3] in vocab_pref_suff:
                  Pref_ID=vocab_pref_suff[word[0:3]]
               if word[-3:] in vocab_pref_suff:
                  Suff_ID = vocab_pref_suff[word[-3:]]
            elif len(word)==2:
                 pref="_"+word
                 suff=word+"_"
                 if pref in vocab_pref_suff:
                    Pref_ID = vocab_pref_suff[pref]
                 if suff in vocab_pref_suff:
                    Suff_ID = vocab_pref_suff[suff]
            elif len(word) == 1:
                 pref = "_" + "_" + word
                 suff = word + "_" + "_"
                 if pref in vocab_pref_suff:
                    Pref_ID = vocab_pref_suff[pref]
                 if suff in vocab_pref_suff:
                    Suff_ID = vocab_pref_suff[suff]
    
           ## Computing case ID here:
           
            case_ID=0
            chars2=list(word)
            case_word="".join([unicodedata.category(c) for c in chars2])
            if case_word in vocab_case:
               case_ID=vocab_case[case_word]
        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                word = vocab_words[UNK]

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids,Pref_ID, Suff_ID,case_ID, word       ############################################### here return prufix_suffix  ## , Pref_ID, Suff_ID
        #elif vocab_pref_suff is not None and Pref_Suff == True:
            #return char_ids, Pref_ID, Suff_ID, word
        else:
            return word

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:          ################# This is for words
        max_length = max(map(lambda x : len(x), sequences))
        global max_words_in_sent
        max_words_in_sent=max_length
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:   ############ This is for Chars
        max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok]*max_length_word,
                                            max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)

    elif nlevels==3:
        pass   ########### This case is for including both suffix and prefix

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)
    Returns:
        list of tuples
    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    global label2ind
    label2ind=idx_to_tag
    #print (idx_to_tag)
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks

def label2ind_ret():
    return label2ind