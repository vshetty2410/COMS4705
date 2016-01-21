from nltk.align.ibm1 import IBMModel1
from nltk.align.ibm2 import IBMModel2
import time

# Initialize IBM Model 1 and return the model.
def create_ibm1(aligned_sents):
    return IBMModel1(aligned_sents, 10)


# Initialize IBM Model 2 and return the model.
def create_ibm2(aligned_sents):
    return IBMModel2(aligned_sents, 10)


# Compute the average AER for the first n sentences
# in aligned_sents using model. Return the average AER.
def compute_avg_aer(aligned_sents, model, n):
    aer = []
    for i in xrange(n):
        aligned_words = model.align(aligned_sents[i])
        aer.append(aligned_sents[i].alignment_error_rate(aligned_words))
    return (0.0 + sum(aer)) / aer.__len__()


# Computes the alignments for the first 20 sentences in
# aligned_sents and saves the sentences and their alignments
# to file_name. Use the format specified in the assignment.
def save_model_output(aligned_sents, model, file_name):
    output_file = open(file_name,'w')
    for aligned_sent in aligned_sents[:20]:
        result = model.align(aligned_sent)
        output_words = str(result.words)  
        output_mots = str(result.mots)
        output_align = str(result.alignment)
        output = output_words+'\n'+output_mots+'\n'+output_align+'\n'+'\n'
        output_file.write(output)
    
    output_file.close()


def main(aligned_sents):
    time.clock()
    ibm1 = create_ibm1(aligned_sents)
    save_model_output(aligned_sents, ibm1, "ibm1.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm1, 50)

    print ('IBM Model 1')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))

    ibm2 = create_ibm2(aligned_sents)
    save_model_output(aligned_sents, ibm2, "ibm2.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm2, 50)

    print ('IBM Model 2')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
    print
    print "Part A time: " + str(time.clock()) + ' sec'