UNI - vs2567
Name - Varun Jagdish Shetty

Question 1-

a)	
Visualization created using the MaltEval.jar with gold standards for English, Danish, Swedish and Korean training data and the results are stored in files-
		1) figure_en.png
		2) figure_da.png
		3) figure_sw.png
		4) figure_ko.png

b)	
A dependency graph is projective if it doesn't have any arcs that cross each other. Visually, if none of the lines overlap with any other line, then it is projective. If there are criss-crossing lines, then it's not projective.
The algorithm to check for projectivity in the 'providedcode/transitionparser.py' file is to iterate through all the arcs. For each arc, if the arc is from i to j, there should be a directed path (using one or more hops) to go from i to every word between i and j.

c)  
Projective sentence: "I went to the hat store and bought a hat."
Non-projective sentence: "I picked up the phone quickly that was distracting the students."

This sentence is non-projective because the arc between "picked" to "quickly" crosses the arc between "phone" and "was".

-------------------------------------------------------------------------------------------------------------------------------------

Question 2-

a)  
Completed configuration of transition.py for the operations left_arc, right_arc, shift, and reduce and implemented test.py with	badfeatures.model for the next sub-question

b)  
Applying badfeatures.model to Swedish test data:

	$ python test.py
	This is not a very good feature extractor!
	UAS: 0.229038040231 
	LAS: 0.125473013344
	It gives us a 12.5% labeled accuracy score, which is very low.

-------------------------------------------------------------------------------------------------------------------------------------

Question 3-

a) 
I added several types of features beyond what's in Table 3.2 in the textbook. I got these features from the paragraphs immediately around the table.

	Feature type 1:

	Distance between the token on top of the stack and the first token in the input buffer.
	It's simply the difference in the address of the two tokens, so it is O(1).

	By adding this, the performance in English increased from

	UAS: 0.676543209877 
	LAS: 0.646913580247

	to

	UAS: 0.698765432099 
	LAS: 0.669135802469

	Linguistically, this feature makes sense because if two words are far away, they are less likely to be related.
	So, distance is a good indicator of dependence relations.

	Feature type 2:

	The number of VERBs that occur between the token on top of the stack and the first token in the input buffer.
	I loop through all the tokens that occur between those two tokens and add up those tagged as VERB. This is O(N).

	By adding this, the performance in English increased from:

	UAS: 0.698765432099 
	LAS: 0.669135802469

	to

	UAS: 0.706172839506 
	LAS: 0.674074074074

	This is the least useful of the features that I added.

	Linguistically, this makes sense because the more verbs there are in between two words, we can think of those words as being "further apart" in some abstract way, so they're less likely to be related. It's as if the more verbs separate two words, the more concepts are introduced in between them, so those two words are less likely to be related.

	Feature type 3:

	The number of children (both left and right, each as a separate feature) of the token on top of the stack,
	as well as the number of children of the first token in the input buffer.
	I loop through the arcs and count the number of arcs where the first index in the 3-tuple is the token of interest. This is O(N).

	By adding this, the performance in English increased from:

	UAS: 0.706172839506 
	LAS: 0.674074074074

	to

	UAS: 0.745679012346
	LAS: 0.718518518519

	This is the most useful of the features that I added.

	Linguistically, this meakes sense because if a word already has dependent words, then it is more likely to be higher up in the dependency parse tree, which means it is more likely to have more dependent words.

b)  
All models trained and stored as follows-
	1) english.model
	2) swedish.model
	4) danish.model

	The swedish model subdata had to sampled with a 223 training examples to get 200 projective sentences for training and the danish model had be sampled with 236 training examples to get 200 projective sentences

c) 	
Top performance by the Swedish and Danish models on the training set data is as under-

	Danish:
	UAS: 0.809780439122
	LAS: 0.730938123752

	Swedish:
	UAS: 0.795658235411
	LAS: 0.685321649074


d.  
The total complexity of the parser is the complexity of the SVM, which is generally between O(N^2) and O(N^3) depending on the algorithm and implementation.
Feature extraction is O(N^2) because there are N tokens, and some features are of order O(N), so O(N*N) = O(N^2).

Parsing a sentence is O(N) in time because the buffer can only decrease in size, which means it's a single-pass parser. However, the upper bound of the space complexity is O(N^2) because in theory, there could be an arc between every token and every other token.

Tradeoffs:

An advantage of this parser is the machine learning technique used can be swapped out with any other technique in the same class, and it can be fine-tuned for the specific application and corpus.

A disadvantage is its inability to parse non-projective sentences. There are techniques to turn non-projective sentences into pseudo-projective sentences, so maybe there are workarounds for this.

-------------------------------------------------------------------------------------------------------------------------------------
Question 3-

a)	
parse.py created which takes in the english.model file as its first argument and the user input sentences are parsed and the appropriate .conll file is generated with the dependency parse tree. There was an missing argument issue with loading the conll file using the MaltEval.jar which was resolved when the whole file path to the output conll file was passed as the argument

