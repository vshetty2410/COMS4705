from main import replace_accented
from sklearn import svm
from sklearn import neighbors
import json
import nltk
import unicodedata
import codecs
import sys
import string

# don't change the window size
window_size = 10

# A.1
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
            lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
            ...
        }
    :return: dic s with the following structure:
        {
            lexelt: [w1,w2,w3, ...],
            ...
        }

    '''
    s = {}
    for key in data:
        final_list = []; left_list = []; right_list = []; left_list_punc = []; right_list_punc = []; sets =  []
        s[key] = []
        for x in data[key]:

            left_context = nltk.word_tokenize(x[1])
            left_list = left_list + (left_context[-window_size:])
            right_context = nltk.word_tokenize(x[3])
            right_list = right_list + right_context[:window_size]
            for word1 in left_list:
                if word1 not in string.punctuation:
                   left_list_punc.append(word1)
            for word2 in right_list:
                if word2 not in string.punctuation:
                   right_list_punc.append(word2)
        final_list = left_list_punc + right_list_punc

        for word_list in final_list:
            if word_list not in s[key]:
                s[key].append(word_list)


    return s


# A.1
def vectorize(data, s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
            [(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}


    for instance in data:
        instance_id = instance[0]
        if instance_id not in vectors:
            vectors[instance_id] = []
        if instance_id not in labels:
            labels[instance_id] = instance[4]
        words_left = nltk.word_tokenize(instance[1])
        words_right = nltk.word_tokenize(instance[3])
        for word in s:
            word_count_left = words_left.count(word)
            word_count_right = words_right.count(word)
            total_count = word_count_left+word_count_right
            vectors[instance_id].append(total_count)

    return vectors, labels

# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''
    svm_results = []; knn_results = []

    svm_clf = svm.LinearSVC()
    knn_clf = neighbors.KNeighborsClassifier()

    training_list = []; sense_list = []
    tup = ()
    for instance in X_train:
        training_list.append(X_train[instance])

    for sense in y_train:
        sense_list.append(y_train[sense])

    svm_clf.fit(training_list,sense_list)
    knn_clf.fit(training_list,sense_list)
    for instance in X_test:
        svm_list =  svm_clf.predict(X_test[instance])
        knn_list = knn_clf.predict(X_test[instance])
        svm_tup = (instance,svm_list[0])
        knn_tup = (instance,knn_list[0])
        svm_results.append(svm_tup)
        knn_results.append(knn_tup)

    return svm_results, knn_results

# A.3, A.4 output
def print_results(results ,output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''

    # implement your code here
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results on instance_id before printing
    outfile = codecs.open(output_file, encoding='utf-8', mode='w')    
    for lexelt, instances in sorted(results.iteritems(), key=lambda d: replace_accented(d[0].split('.')[0])):
        for instance in sorted(instances, key=lambda d: int(d[0].split('.')[-1])):
            instance_id = instance[0]
            sid = instance[1]
            outfile.write(replace_accented(lexelt + ' ' + instance_id + ' '  + sid  + '\n'))
    outfile.close()

# run part A
def run(train, test, language, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)
    
    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)