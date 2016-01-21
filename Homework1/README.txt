UNI - vs2567
Name - Varun Jagdish Shetty

Question A-1- output/A1.txt results

UNIGRAM natural -13.766408817
BIGRAM natural that -4.05889368905
TRIGRAM natural that he -1.58496250072

Question A-2- output/A2.txt results

Unigram Model perplexity
$ python perplexity.py output/A2.uni.txt data/Brown_train.txt
The perplexity is 1052.4865859

Bigram Model perplexity
$ python perplexity.py output/A2.bi.txt data/Brown_train.txt
The perplexity is 53.8984761198

Trigram Model perplexity
$ python perplexity.py output/A2.tri.txt data/Brown_train.txt
The perplexity is 5.7106793082

Question A-3- output/A3.txt results

Linear Interpolation Model perplexity
$ python perplexity.py output/A3.txt data/Brown_train.txt
The perplexity is 12.5516094886

Question A-4- Result analysis of values from Q-3 when compared to Q-2

The expected result ideally should have been the linear interpolation model offering a better fit to the training data(Brown_train.txt) given that all three models are simultaneously applied but the case isn't so with the trigram model performing a bit better than it with regards to the perplexity offered by both models. I believe the reason for this is, as we can see from the formula, the sum of uni,bi,tri_gram probability is divided by three and not just the sum of the three values.  Also, a case where, given the non-existence of a trigram and the corresponding bigram there exists a unigram score for the token , the token would have a score value being appended which in turn would give rise to a higher score and I believe a higher perplexity. 

Question A-5-

	Section 1-

	Perplexity of Sample1_scored.txt against the Sample1.txt

	$ python perplexity.py output/Sample1_scored.txt data/Sample1.txt
	The perplexity is 11.1670289158

	Perplexity of Sample2_scored.txt against the Sample2.txt

	$ python perplexity.py output/Sample2_scored.txt data/Sample2.txt
	The perplexity is 1611240282.44

	Section 2-

	The Sample1 has a perplexity value ~11 which is much closer to 1 when compared to the perplexity value offered by the Sample2 (~1611240282) which leads me to conclude that the Sample1 belongs to the Brown dataset as the distribution of the n-grams in the Sample1 is much closer to the distribution of n-grams in the training data.



Assignment Part B- Part of Speech Tagging

Question B-2- output/B2.txt[calc_trigrams()] results

TRIGRAM CONJ ADV ADP -2.9755173148
TRIGRAM DET NOUN NUM -8.9700526163
TRIGRAM NOUN PRT PRON -11.0854724592

Question B-4- output/B4.txt[calc_emission()] results

* * 0.0
Night NOUN -13.8819025994
Place VERB -15.4538814891
prime ADJ -10.6948327183
STOP STOP 0.0
_RARE_ VERB -3.17732085089

Question B-5- output/B5.txt [viterbi()] results

$ python pos.py output/B5.txt data/Brown_tagged_dev.txt
Percent correct tags: 93.3226493638

Question B-6- output/B6.txt [nltk_tagger()] results

$ python pos.py output/B6.txt data/Brown_tagged_dev.txt
Percent correct tags: 87.9985146677


Code Runtimes-

Part A takes-
$ python solutionsA.py
Part A time: 18.21 sec

Part B takes-
$ python solutionsB.py
Part B time: 552.82 sec
