import nltk
import math
import time
from collections import Counter

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# Function that converts a regular probability to a log probability
def stat_prob_to_log_prob(nonlog_prob):
    if nonlog_prob == 0:
        return float(LOG_PROB_OF_ZERO)
    else:
        return math.log(nonlog_prob, 2)

# TODO: IMPLEMENT THIS FUNCTION-Finished
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []

    # Adding STOP_SYMBOL and 2 START_SYMBOLs to the resultant word and tag lists
    for sent in brown_train:
        sent_list = sent.split() 
        curr_words = [START_SYMBOL, START_SYMBOL]
        curr_tags = [START_SYMBOL, START_SYMBOL]
        for pair in sent_list:
            # Final occurrence of '/' in the pair to take care of the special case mentioned in the guidelines
            slash_index = pair.rfind("/")
            curr_tags.append(pair[slash_index + 1:])
            curr_words.append(pair[:slash_index])

        curr_words.append(STOP_SYMBOL)
        curr_tags.append(STOP_SYMBOL)

        brown_words.append(curr_words)
        brown_tags.append(curr_tags)
        
    return brown_words, brown_tags

# TODO: IMPLEMENT THIS FUNCTION-Finished
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}

    bigram_counts = {}
    trigram_counts = {}
    
    for taglist in brown_tags:
        # Iterating at index 1 for bi,tri- grams.
        for i in xrange(1, len(taglist)):
            if i != 1:
                # look at the previous 2 words and the current word
                trigram = (taglist[i-2], taglist[i-1], taglist[i])
                # check in q_values -- if the tuple isn't there, set value to 1, otherwise increment
                trigram_counts[trigram] = trigram_counts.get(trigram, 0) + float(1)

            # look at the previous word and the current word
            bigram = (taglist[i-1], taglist[i])
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + float(1)


    # the probability is the trigram count divided by the bigram count of the first 2 words
    for trigram, tricount in trigram_counts.iteritems():
        probability = float(tricount) / bigram_counts[(trigram[0], trigram[1])]
        q_values[trigram] = stat_prob_to_log_prob(probability)
    
    return q_values


# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION-Finished
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = []

    # Building a counter of all the words
    count_dict = Counter([word for sublist in brown_words for word in sublist])

    # Iterating through the Counter object and appending to known_words if >5 occurrences
    for word, count in count_dict.iteritems():
        if count > RARE_WORD_MAX_FREQ:
            known_words.append(word)

    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []

    # Creating a set for faster access
    known_set = set(known_words)

    # Iterating through brown and making the replacement
    for sent in brown_words:
        processed_sent = []
        for word in sent:
            if word in known_set:
                processed_sent.append(word)
            else:
                processed_sent.append(RARE_SYMBOL)
        brown_words_rare.append(processed_sent)
    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION-Finished
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = []
 
    # Counting each tag and building taglist
    tag_counts = Counter([tag for sublist in brown_tags for tag in sublist])
    taglist = tag_counts.keys()

    # Counting the co-occurrence pairs -- this dict is mapping (word, tag) ==> count
    pair_counts = {}
    for i in xrange(len(brown_words_rare)): # iterate over sentences
        sent = brown_words_rare[i]
        for j in xrange(len(sent)): # iterate over words/tags
            word_tag_tuple = (sent[j], brown_tags[i][j])
            # set count to 1 if it's the first occurrence, otherwise increment
            pair_counts[word_tag_tuple] = pair_counts.get(word_tag_tuple, 0) + float(1)
    
    # Building e_values using pair_counts and tag_counts
    for tuple, pair_count in pair_counts.iteritems():
        e_values[tuple] = stat_prob_to_log_prob( float(pair_count) / float(tag_counts[tuple[1]]) )

    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION-Finished
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []

    taglist_set = set(taglist)
     # Deleting the * and STOP tags since we'll never actually tag anything using them. Saw a bump up in the processing by ~300 seconds
    taglist_set.remove(START_SYMBOL)
    taglist_set.remove(STOP_SYMBOL)
    taglist = list(taglist_set)

    known_words_set = set(known_words)

    count = 0
    for token_list in brown_dev_words:
        count += 1
        
        token_list.insert(0, "DUMMY_INDEX_ONE")
        
        word_tag_list = []
        pi = {}
        backpointers = {}

        for k in xrange(1, len(token_list)): # Trigram HMM, so we'll be looking at i-2. start at 2.
            curr_token = token_list[k]

            # Handling rare words
            if curr_token not in known_words_set:
                curr_token = RARE_SYMBOL

            possible_two_prev_tags = taglist_for_token_index(taglist, k-2)
            possible_prev_tags = taglist_for_token_index(taglist, k-1)
            possible_curr_tags = taglist_for_token_index(taglist, k)

            for u in possible_prev_tags:
                for v in possible_curr_tags:
                    max_prob_seen = -4000.0

                    # Iterating over possible tags 2 tokens ago and picking the one that maximizes the current probability
                    for w in possible_two_prev_tags:
                        curr_prob = pi_for_token_index(pi, k-1, w, u) + q_values.get((w, u, v), LOG_PROB_OF_ZERO) + e_values.get((curr_token, v), LOG_PROB_OF_ZERO)
                        
                        # either this is the first w we're seeing, or it's the best one. either way, storing it.
                        if curr_prob >= max_prob_seen or ((k, u, v) not in backpointers):
                            max_prob_seen = curr_prob

                            # Storing the best (or first) w in the backpointer dict, along with its probability
                            backpointers[(k, u, v)] = w
                            pi[(k, u, v)] = curr_prob

        # n is for bookkeeping, y will be the list of tags
        n = len(token_list)-1
        y = [None] * (n+1)

        possible_last_tags = taglist_for_token_index(taglist, n)
        possible_prev_to_last_tags = taglist_for_token_index(taglist, n-1)

        max_last_prob = -4000.00

        # Finding the best possible tags to end the sentence
        for u in possible_prev_to_last_tags:
            for v in possible_last_tags:
                curr_prob = pi_for_token_index(pi, n, u, v) + q_values.get((u, v, STOP_SYMBOL), LOG_PROB_OF_ZERO)

                if curr_prob >= max_last_prob:
                    max_last_prob = curr_prob
                    
                    y[len(y)-1] = v
                    y[len(y)-2] = u

        # walk backwards from the last two tokens (which we've tagged) and get the tags we put into backpointers
        for k in reversed(xrange(1, n-1)):
            y[k] = backpointers[(k+2, y[k+1], y[k+2])]

        # match each token with its tag
        for i in xrange(1, len(token_list)):
            word_tag_list.append(token_list[i] + "/" + y[i])
            
        # join all the WORD/TAGs together and put a newline at the end
        tagged.append(" ".join(word_tag_list) + "\n")

    return tagged



# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION-Finished
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    tagged = []

    # Thank you for this
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]

    # train the nltk taggers
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)

    for token_list in brown_dev_words:
        # first, tag the tokens
        tagged_tuples = trigram_tagger.tag(token_list)

        # now, format the tagged tuples into strings
        sent_output = []
        for tag_tuple in tagged_tuples:
            sent_output.append(tag_tuple[0] + "/" + tag_tuple[1])

        tagged.append(sent_output)
    
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        output = ' '.join(sentence) + '\n'
        outfile.write(output)
    outfile.close()


# Helper function for Viterbi that returns the possible list of tags given
# the entire taglist and which token we're on
def taglist_for_token_index(taglist, token_index):
    if token_index < 1:
        return [START_SYMBOL]
    else:
        return taglist

# Helper function for Viterbi that takes the path probabilities,
# the token we're on, and the 2 tags, and returns the probability
def pi_for_token_index(pi, token_index, tag1, tag2):
    if token_index == 0 and tag1 == START_SYMBOL and tag2 == START_SYMBOL:
        return 0.0 # ie prob(nonlog) = 1
    else:
        return pi.get((token_index, tag1, tag2), LOG_PROB_OF_ZERO)

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    #open Brown training data
    infile = open(DATA_PATH+"Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    #split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    #calculate trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    #question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    #calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    #get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    #question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    #calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    #question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    #open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    #format Brown development data here
    brown_dev_words = []
    brown_dev_words_nltk= []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])
        brown_dev_words_nltk.append(sentence.split(" ")[:-1])

    #do viterbi on brown_dev (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    #question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')
    

    #do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words_nltk)
    
    #question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()