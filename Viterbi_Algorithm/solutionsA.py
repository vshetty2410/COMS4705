import math
import nltk
import time
from collections import Counter

# Constants to be used by you when you fill the functions
START_SYMBOL = '* '
STOP_SYMBOL = ' STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# Function that converts a regular probability to a log probability
def stat_prob_to_log_prob(stat_prob):
    if stat_prob == 0:
        return float(MINUS_INFINITY_SENTENCE_LOG_PROB)
    else:
        return math.log(stat_prob, 2)

# Function that converts a log probability to non-log
def log_prob_to_stat_prob(log_prob):
    if log_prob == float(MINUS_INFINITY_SENTENCE_LOG_PROB):
        return float(0)
    else:
        return math.pow(2, log_prob)
    

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# brown(training_corpus): is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(brown):

    # Iterate through brown, add STOP at the end of each sentence
    brown_with_stop = []
    for brown_sent in brown:
        brown_with_stop.append(brown_sent.rstrip() + STOP_SYMBOL)

    # Total number of sentences
    sentence_count = float(len(brown))

    # Make the whole text just one string for easy manipulation
    full_brown_text_with_stop = "\n".join(brown_with_stop)

    # Bulding the unigram model by first tokenizing the entrie text
    tokens_with_stop = full_brown_text_with_stop.strip().split()

    # Number of tokens including STOPs. Required later.
    token_count = float(len(tokens_with_stop))

    # Counter is a subclass of dict from collections. Helps in really fast computations for lists
    unigram_p_counts = Counter(tokens_with_stop)

    # Building unigram model
    unigram_p = {(unigram,) : stat_prob_to_log_prob(unigram_p_counts[unigram] / token_count) for unigram in unigram_p_counts}

     # Iterate through brown, add * at the start of each sentence
    brown_with_start_stop = []
    for brown_sent in brown_with_stop:
        brown_with_start_stop.append(START_SYMBOL + brown_sent)

    # Make it all one big text for manipulation again
    full_brown_text_with_start_stop = "\n".join(brown_with_start_stop)

    # Creating bigram model
    bigram_tuples = tuple(nltk.bigrams(full_brown_text_with_start_stop.strip().split()))

    # Removing all bigrams that end in * from the list created 
    bigram_tuples_clean = []
    for bigram_tuple in bigram_tuples:
        if bigram_tuple[1] != "*":
            bigram_tuples_clean.append(bigram_tuple)

    # loop through brown, add another * at the start of each sentence
    brown_with_startstart_stop = []
    for brown_sent in brown_with_start_stop:
        brown_with_startstart_stop.append(START_SYMBOL + brown_sent)

    # make it all just one string
    full_brown_text_with_startstart_stop = "\n".join(brown_with_startstart_stop)

    # Creating trigram model
    trigram_tuples = tuple(nltk.trigrams(full_brown_text_with_startstart_stop.strip().split()))

    # Removing all trigrams that end in * or *,*
    trigram_tuples_clean = []
    for trigram_tuple in trigram_tuples:
        if trigram_tuple[2] != "*":
            trigram_tuples_clean.append(trigram_tuple)

    # Convert all counts into log-probabilities

    # Building the actual bigram model
    bigram_p_counts = Counter(bigram_tuples_clean)

    # The unclean list includes counts of bigrams that end in *
    bigram_p_counts_unclean = Counter(bigram_tuples)
    bigram_p = {}
    for bigram in bigram_p_counts:
        if unigram_p_counts[bigram[0]] == 0:
            
            if bigram[0] == "*": # sentence start - use sentence count in denominator
                bigram_p[bigram] = stat_prob_to_log_prob(bigram_p_counts[bigram] / sentence_count)
            else:
                bigram_p[bigram] = float(MINUS_INFINITY_SENTENCE_LOG_PROB)
        else:
           # Best case scenario
            bigram_p[bigram] = stat_prob_to_log_prob(float(bigram_p_counts[bigram]) / unigram_p_counts[bigram[0]])
        
    # build the actual trigram model
    trigram_p_counts = Counter(trigram_tuples_clean)
    trigram_p = {}
    for trigram in trigram_p_counts:
        if bigram_p_counts_unclean[(trigram[0],trigram[1])] == 0:
            if trigram[0] == "*" and trigram[1] == "*": # sentence start - use sentence count in denominator
                trigram_p[trigram] = stat_prob_to_log_prob(trigram_p_counts[trigram] / sentence_count)
            else:
                trigram_p[trigram] = float(MINUS_INFINITY_SENTENCE_LOG_PROB)
        else:
            # Best case scenario
            trigram_p[trigram] = stat_prob_to_log_prob(float(trigram_p_counts[trigram]) / bigram_p_counts_unclean[(trigram[0],trigram[1])])
    
    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams,filename):
    #output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')
    
    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')
    
    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()  
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')
    
    outfile.close()
    
# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# data(corpus): list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, data):

    scores = []

    # Iterate over sentences, rstrip(), append STOP, prepend n-1 asterisks
    data_clean = []
    asters = ""
    for i in range(n-1): # To append the optimal number of asterisks per model
        asters = asters + START_SYMBOL

    for sent in data:
        data_clean.append(asters + sent.rstrip() + STOP_SYMBOL)

     # For each sentence, tokenize. split into ngrams if n > 1 and create list of tuples
    data_tokens = []
    data_tuples = []
    for sent in data_clean:
        tokens = sent.split()
        data_tokens.append(tokens)
        if n == 1:
            tuples = [(token,) for token in tokens]
            data_tuples.append(tuples)
        elif n == 2:
            data_tuples.append(tuple(nltk.bigrams(tokens)))
        elif n == 3:
            data_tuples.append(tuple(nltk.trigrams(tokens)))
    
  # For each sentence, add (logspace) or multiply (non-log) scores from ngram_p together for all ngrams or tokens
    for sent in data_tuples:
        score = 0.0
        for curr_tuple in sent:
            if ngram_p.get(curr_tuple, float(MINUS_INFINITY_SENTENCE_LOG_PROB)) == float(MINUS_INFINITY_SENTENCE_LOG_PROB):
                # We've found a tuple that's not in our model. we can set the score to -1000 and break out of this sentence
                score = float(MINUS_INFINITY_SENTENCE_LOG_PROB)
                break
            
            # Best case scenario
            score = score + ngram_p[curr_tuple]

           
        scores.append(score)
        
    return scores


# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, brown):

    scores = []

    # lambda l for all 3 models is 1/3
    l = float(1)/3

    # Clean up data by chopping off \r\n from the end, appending STOP
    data_clean = [sent.rstrip() + STOP_SYMBOL for sent in brown]

    # this list will contain lists of tokens
    tokenized_sentences = []
    for sent in data_clean:
        tokens = sent.split()
        tokenized_sentences.append(tokens)

    zero_score = float(MINUS_INFINITY_SENTENCE_LOG_PROB)

    # Iterate over list of sentences
    for tokens in tokenized_sentences:
        score = 0.0
        for i in xrange(len(tokens)):
            # Build the tuples we need for the model lookups
            unigram_tuple = (tokens[i],)

            bigram_list = []
            trigram_list = []

            # These if-statement handle the case when we're at the start of a sentence. 
            if i-2 < 0:
                trigram_list.append("*")
            else:
                trigram_list.append(tokens[i-2])

            if i-1 < 0:
                trigram_list.append("*")
                bigram_list.append("*")
            else:
                trigram_list.append(tokens[i-1])
                bigram_list.append(tokens[i-1])

            trigram_list.append(tokens[i])
            bigram_list.append(tokens[i])

            bigram_tuple = tuple(bigram_list)
            trigram_tuple = tuple(trigram_list)

            # Get the scores from each of the models

            uniscore = unigrams.get(unigram_tuple, zero_score)
            biscore = bigrams.get(bigram_tuple, zero_score)
            triscore = trigrams.get(trigram_tuple, zero_score)

            # At least one of the n-grams is not in the models, so zero-out the score and move on to the next sentence
            if uniscore == zero_score and biscore == zero_score and triscore == zero_score:
                score = zero_score
                break

            # Interpolation formula
            nonlog_score = l * (log_prob_to_stat_prob(uniscore) + log_prob_to_stat_prob(biscore) + log_prob_to_stat_prob(triscore))
            score = score + stat_prob_to_log_prob(nonlog_score)

        scores.append(score)
            
    
    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    #question 3 output
    score_output(linearscores, OUTPUT_PATH + '/A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    #score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    #question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()