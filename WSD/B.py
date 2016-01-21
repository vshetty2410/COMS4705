import string
from nltk.tag.brill import nltkdemo18
import A
from sklearn.feature_extraction import DictVectorizer
import nltk
from nltk.corpus import stopwords, cess_esp, cess_cat
from nltk.stem.snowball import SnowballStemmer
from sklearn import svm
import math
import operator


# You might change the window size
window_size = 10

#_POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
#tagger = nltk.load(_POS_TAGGER)

high_frequency_words = {}

# def calc_high_frequency_words(data):
#     for lexelt in data:
#         dict_count = {}
#         dict_word_count = {}
#         top_list = {}
#         for instance in data[lexelt]:
#             sense_id = instance[4]
#             left_context = nltk.word_tokenize(instance[1])
#             left_context = left_context[-window_size:]
#             right_context = nltk.word_tokenize(instance[3])
#             right_context = right_context[:window_size]

#             for left_words in left_context:
#                 if left_words in dict_count and left_words not in string.punctuation:
#                     dict_count[left_words] += 1
#                 else:
#                     dict_count[left_words] = 1

#             for right_words in right_context:
#                 if right_words in dict_count and right_words not in string.punctuation:
#                     dict_count[right_words] += 1
#                 else:
#                     dict_count[right_words] = 1
#             for left_words in left_context:
#                 if sense_id in dict_word_count:
#                     if left_words in dict_word_count[sense_id] and left_words not in string.punctuation:
#                         dict_word_count[sense_id][left_words] +=1
#                     else:
#                         dict_word_count[sense_id][left_words] = 1
#                 else:
#                     dict_word_count[sense_id] = {left_words:1}

#             for right_words in right_context:
#                 if sense_id in dict_word_count:
#                     if right_words in dict_word_count[sense_id] and right_words not in string.punctuation:
#                         dict_word_count[sense_id][right_words] +=1
#                     else:
#                         dict_word_count[sense_id][right_words] = 1
#                 else:
#                     dict_word_count[sense_id] = {right_words:1}

#         rank = 0
#         for sense_id in dict_word_count:
#             for word in dict_word_count[sense_id]:
#                 N_sc = dict_word_count[sense_id][word]
#                 N_c = dict_count[word]
#                 N_c_bar = float(abs(N_sc-N_c))
#                 psc = float(float(N_sc)/float(N_c))

#                 pscBar = float(N_c_bar)/float(N_c)
#                 if pscBar == 0.0:
#                     rank = 0
#                 else:
#                     rank = float(math.log(float(psc)/float(pscBar),2))

#                 if word in top_list:
#                     if top_list[word] < rank:
#                         top_list[word] = rank
#                 else:
#                     top_list[word] = rank

#         top_list = sorted(top_list.items(), key=operator.itemgetter(1), reverse=True)

#         i = 0
#         word_list = []
#         for topword in top_list:
#             word_list.append(topword[0])
#             i+=1
#             if i>7:
#                 break
#         high_frequency_words[lexelt] = word_list

# B.1.a,b,c,d
def extract_features(data,language,lexelt,s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
            [(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''
    i = 1; 
    print 'Before feature extraction'
    features = {}; vectors = {};  labels = {}; 
    left_context = [];  left_list = [];  right_list = []; right_context = []
    # if language.lower() != 'catalan':
    #     stop_words = stopwords.words(language.lower())
    #     stemmer = SnowballStemmer(language.lower(), ignore_stopwords=True)
    # else:
    #     stop_words = []
    #     stemmer = []
    # count = 0

    

    for instance in data:
        labels[instance[0]] = instance[4]
        sense_count = {}
 
        # left_context = nltk.word_tokenize(instance[1])
        # right_context = nltk.word_tokenize(instance[3])

        #pos_list  = pos_tagger(left_context, right_context,language.lower())
        final_dict = {}
        # for i in range(0,len(pos_list)):
        #       final_dict["POS" + str(i-2)] = pos_list[i]


   
        instance_id = instance[0]
        if instance_id not in vectors:
            vectors[instance_id] = []
        words_left = nltk.word_tokenize(instance[1])
        words_right = nltk.word_tokenize(instance[3])
        
        for word in s:
            word_count_left = words_left.count(word)
            word_count_right = words_right.count(word)
            total_count = word_count_left+word_count_right
            final_dict["W" + str(i) ] = total_count
            i = i + 1


        # left_stemmer_list = []; right_stemmer_list = []; final_left_stemmer_list = []; final_right_stemmer_list = []; final_stemmer_list = []
        # for left in left_context:
        #     if (left  not in stop_words and (language.lower() == 'english' or language.lower() == 'spanish')) and  left not in string.punctuation :
        #         left_stemmer_list.append(stemmer.stem(left.lower()))

        #     elif language.lower() == 'catalan':
        #         left_stemmer_list.append(left)
        # final_left_stemmer_list.append(left_stemmer_list[-window_size:])


        # for right in right_context:
        #     if (right not in stop_words and (language.lower() == 'english' or language.lower() == 'spanish')) and right not in string.punctuation :
        #         right_stemmer_list.append(stemmer.stem(right.lower()))
        #     elif language.lower() == 'catalan':
        #         right_stemmer_list.append(right)

        # final_right_stemmer_list.append(right_stemmer_list[:window_size])

        # context_list1 = []
        # context_list2 = []
        # final_stemmer_list = final_left_stemmer_list + final_right_stemmer_list
        # for left in left_context:
        #     if left not in string.punctuation:
        #         context_list1.append(left)
        # for right in right_context:
        #     if right not in string.punctuation:
        #         context_list2.append(right)
        # context_list = context_list1 + context_list2

        # for top_word in context_list:
        #     if top_word in high_frequency_words[lexelt]:
        #         if "rel_"+top_word in final_dict:
        #             final_dict["rel_" + top_word] += 1
        #         else:
        #             final_dict["rel_" + top_word] = 1

        features[instance[0]] = final_dict

        # final_list = []
        # for word in final_stemmer_list:
        #     for x in word:
        #         final_list.append(x)


        # word_set = set([])
        # for word in final_list:
        #     synonim_list = []
        #     hyponyms_list = []
        #     hypernyms_list = []

        #     word_set.add(word)
        #     synsets = nltk.corpus.wordnet.synsets(word)
        #     for x in synsets:
        #         synonim_list.append(x.name().split('.')[0])
        #     for syn in synonim_list:
        #         word_set.add(syn)



        #     for x in xrange(0,len(synonim_list)):
        #         if synonim_list[x] == word:
        #             hyponyms = synsets[x].hyponyms()
        #             for i in hyponyms:
        #                 hyponyms_list.append(i.name().split('.')[0])
        #             for hyp in hyponyms_list:
        #                 word_set.add(hyp)
        #             hypernyms = synsets[x].hypernyms()
        #             for i in hypernyms:
        #                 hypernyms_list.append(i.name().split('.')[0])
        #             for hypr in hypernyms_list:
        #                 word_set.add(hypr)

        #     if instance[4] in sense_count:
        #         sense_count[instance[4]] += 1
        #     else:
        #         sense_count[instance[4]] = 1


        # final_list = list(word_set)

        # for list_words in final_list:
        #     if "hyp_"+list_words in final_dict:
        #         final_dict["hyp_"+list_words] +=1
        #     else:
        #         final_dict["hyp_"+list_words] = 1

    #print features

    #print final_left_stemmer_list[-window_size:]
    print 'Feature Extracted'
    return features, labels


# def pos_tagger(left_context, right_context,language):

#     left_context = left_context[-3:]
#     right_context = right_context[:3]
#     tagger_list = left_context + right_context
#     result = []
#     if language == 'english':
#         tagger_list = tagger.tag(tagger_list)
#         for item in tagger_list:
#             result.append(item[1])
#     elif language == 'spanish':
#         training = cess_esp.tagged_sents()
#         default_tagger = nltk.DefaultTagger('NN')
#         unigram_tagger = nltk.UnigramTagger(training,backoff=default_tagger)
#         bigram_tagger = nltk.BigramTagger(training, backoff=unigram_tagger)
#         esp_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
#         tagger_list = tagger.tag(tagger_list)
#         for item in tagger_list:
#             result.append(item[1])
#     else:
#         training = cess_cat.tagged_sents()

#         default_tagger = nltk.DefaultTagger('NN')
#         unigram_tagger = nltk.UnigramTagger(training,backoff=default_tagger)
#         bigram_tagger = nltk.BigramTagger(training, backoff=unigram_tagger)
#         esp_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
#         tagger_list = tagger.tag(tagger_list)
#         for item in tagger_list:
#             result.append(item[1])

#     return result


# implemented for you
def vectorize(train_features,test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}
    print 'Vectorize start'
    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]
    print 'Vectorize end'
    return X_train, X_test

#B.1.e
def feature_selection(X_train,X_test,y_train,language):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''


    # X_train_new = {}
    # X_test_new = {}
    # X = []
    # Y = []
    # T = []
    # for x in X_train:
    #     X.append(X_train[x])
    #
    #
    # for y in y_train:
    #     Y.append(y_train[y])
    #
    # for t in X_test:
    #     T.append(X_test[t])
    #
    #
    # feature_selector = SelectKBest(chi2, k=10).fit(X,Y)
    # X_test_new = feature_selector.transform(T)
    # x_train_new = feature_selector.transform(X)
    #

    return X_train, X_test

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

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

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''
    svm_results = []; 
    svm_clf = svm.LinearSVC()
    training_list = []; sense_list = []
    tup = ()
    for instance in X_train:
        training_list.append(X_train[instance])

    for sense in y_train:
        sense_list.append(y_train[sense])

    svm_clf.fit(training_list,sense_list)
    
    for instance in X_test:
        svm_list =  svm_clf.predict(X_test[instance])
        svm_tup = (instance,svm_list[0])
        svm_results.append(svm_tup)

    return svm_results

    # implement your code here

    #return results

# run part B
def run(train, test, language, answer):
    results = {}
    #calc_high_frequency_words(train)
    print 'Calling A'
    s = A.build_s(train)
    for lexelt in train:

        train_features, y_train = extract_features(train[lexelt],language,lexelt,s[lexelt])
        test_features, _ = extract_features(test[lexelt],language,lexelt,s[lexelt])

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train,language)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)
    A.print_results(results, answer)
    print 'ended'