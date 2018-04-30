import json
import codecs
from keras.preprocessing.sequence import pad_sequences
import nltk
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import validator
from keras.utils import to_categorical
from nltk.stem import *
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

#Multilayer Perceptron (MLP) for multi-class softmax classification:

def feed_forward(x_train, y_train, x_test, y_test):
    BATCHSIZE = 50
    ITERATIONS = 100
    INPUT_SIZE = len(x_train[0])
    OUTPUT_SIZE = len(validator.EN_SENSES)

    # Dense() is a fully-connected layer with 32 nodes.
    # in the first layer, you must specify the expected input data shape
    print("building model...\n")
    model = Sequential()
    model.add(Dense(units=32, activation='tanh', input_dim=INPUT_SIZE))
    model.add(Dropout(0.5))
    model.add(Dense(units=32, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(units=OUTPUT_SIZE, activation='softmax'))

    print("computing loss...\n")
    # sgd = SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print("fitting model....\n")
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    model.fit(x_train, y_train, epochs=ITERATIONS, batch_size=BATCHSIZE, verbose=2)

    print("scoring model...\n")
    score = model.evaluate(x_test, y_test, batch_size=BATCHSIZE)
    print(score)

    print("predicting senses...\n")
    predict = model.predict(x_test, batch_size=BATCHSIZE)
    print("Done!\n")
    return predict


def process_data(dev_relations, training_relations, test_relations):
    """
    This function takes the dev, training and test data and converts it to
    word embeddings and matching senses, preprocessing the data to be fed
    into the feed-forward neural network.
    :param dev_relations:
    :param training_relations:
    :param test_relations:
    :return: 6-tuple with training, test and dev data for x and y
    """
    stemmer = PorterStemmer()
    #create a sentence bank of all sentences from the raw text in all the json files
    all_relations = dev_relations + training_relations + test_relations
    sentence_bank = []
    for relation in all_relations:
        sentence_bank.append(relation["Arg1"]["RawText"] + " " + relation["Connective"]["RawText"] + " " + relation["Arg2"][
            "RawText"])

    wordlist = []
    for sentence in sentence_bank:
        sentence = nltk.word_tokenize(sentence)
        for word in sentence:
            word = word.lower().strip()
            word = stemmer.stem(word)
            wordlist.append(word)

    #map each word in the vocabulary to an index
    vocab = {word: i + 1 for i, word in enumerate(set(wordlist))}
    vocab["UNK"] = 0
    sentence_bank = [[stemmer.stem(word.lower().strip()) for word in nltk.word_tokenize(sentence)] for sentence in sentence_bank]

    #create dual lists of sentences of raw text and their senses, filtering out ill-formed data
    #for dev, training and test sets
    dev_sentences = []
    dev_senses = []
    for relation in dev_relations:
        if relation["Sense"][0] not in validator.EN_SENSES:
            continue
        else:
            dev_senses.append(relation["Sense"][0])
            dev_sentences.append(relation["Arg1"]["RawText"] + " " + relation["Connective"]["RawText"] +
                                      " " + relation["Arg2"]["RawText"])
    dev_sentences = [[stemmer.stem(word.lower().strip()) for word in nltk.word_tokenize(sentence)] for sentence in dev_sentences]

    training_sentences = []
    training_senses = []
    for relation in training_relations:
        if relation["Sense"][0] not in validator.EN_SENSES:
            continue
        else:
            training_senses.append(relation["Sense"][0])
            training_sentences.append(relation["Arg1"]["RawText"] + " " + relation["Connective"]["RawText"] +
                                        " " + relation["Arg2"]["RawText"])
    training_sentences = [[stemmer.stem(word.lower().strip()) for word in nltk.word_tokenize(sentence)] for sentence in training_sentences]

    test_sentences = []
    test_senses = []
    for relation in test_relations:
        if relation["Sense"][0] not in validator.EN_SENSES:
            continue
        else:
            test_senses.append(relation["Sense"][0])
            test_sentences.append(relation["Arg1"]["RawText"] + " " + relation["Connective"]["RawText"] +
                                  " " + relation["Arg2"]["RawText"])
    test_sentences = [[stemmer.stem(word.lower().strip()) for word in nltk.word_tokenize(sentence)] for sentence in test_sentences]

    #process the labels for the y sets of data
    #convert a list of labels to one-hot vectors
    labels = np.array([i for i, sense in enumerate(validator.EN_SENSES)])
    LEN_LABELS = len(labels)
    one_hot_y = np.array([to_categorical(i, num_classes=LEN_LABELS) for i in labels])
    label2onehot = dict(zip(validator.EN_SENSES,one_hot_y))

    #get list of one hot vectors corresponding to idx of sense for each sentence
    #do mapping here
    train_embeddings_list_x = [[vocab[word] for word in sentence] for sentence in training_sentences]
    test_embeddings_list_x = [[vocab[word] for word in sentence] for sentence in test_sentences]
    dev_embeddings_list_x = [[vocab[word] for word in sentence] for sentence in dev_sentences]

    #padding short x sentences for train, test and dev
    max_len = max([len(sentence) for sentence in sentence_bank])
    train_embeddings_list_x = pad_sequences(train_embeddings_list_x, maxlen=max_len, value=-1., padding="post", truncating="post")
    test_embeddings_list_x = pad_sequences(test_embeddings_list_x, maxlen=max_len, value=-1., padding="post", truncating="post")
    dev_embeddings_list_x = pad_sequences(dev_embeddings_list_x, maxlen=max_len, value=-1., padding="post", truncating="post")

    #get lists of senses for train, test, dev sets
    train_embeddings_list_y = [label2onehot[sense] for sense in training_senses if sense in label2onehot]
    test_embeddings_list_y = [label2onehot[sense] for sense in test_senses if sense in label2onehot]
    dev_embeddings_list_y = [label2onehot[sense] for sense in dev_senses if sense in label2onehot]

    return (train_embeddings_list_x, test_embeddings_list_x, train_embeddings_list_y,
            test_embeddings_list_y, dev_embeddings_list_x, dev_embeddings_list_y)


def main():
    dev_set = codecs.open('dev/relations.json', encoding='utf8')
    dev_relations = [json.loads(x) for x in dev_set]

    training_set = codecs.open('train/relations.json', encoding='utf8')
    training_relations = [json.loads(x) for x in training_set]

    test_set = codecs.open('test/relations.json', encoding='utf8')
    test_relations = [json.loads(x) for x in test_set]

    print("processing data...\n")
    data = process_data(dev_relations, training_relations, test_relations)
    train_x = data[0]
    test_x = data[1]
    train_y = data[2]
    test_y = data[3]
    dev_x = data[4]
    dev_y = data[5]

    results = feed_forward(np.array(train_x), np.array(train_y),
                          np.array(test_x), np.array(test_y))

    # for every onehot in results, convert it back to its label by index
    onehot2label = [validator.EN_SENSES[np.argmax(onehot)] for onehot in results]

    # rewrite data into json format for scorer
    with open('output.json', 'w') as outfile:
        for idx, label in enumerate(onehot2label):
            newRel = {"Arg1": {'TokenList':[]},
                       "Arg2": {'TokenList':[]},
                       'Connective': {'TokenList':[]},
                       'DocID': '',
                       'Sense': '',
                       'Type': ''}
            tr = test_relations[idx]
            newRel['Arg1']['TokenList'] = [list[2] for list in tr['Arg1']['TokenList']]
            newRel['Arg2']['TokenList'] = [list[2] for list in tr["Arg2"]["TokenList"]]
            newRel['Connective']['TokenList'] = [item for sublist in tr["Connective"]["TokenList"]
                                                 for item in sublist] if tr["Type"] != "Implicit" else []
            newRel['DocID'] = tr['DocID']
            newRel['Sense'] = [label]
            newRel['Type'] = tr['Type']
            json.dump(newRel, outfile)
            outfile.write('\n')

if __name__ == '__main__':
    main()