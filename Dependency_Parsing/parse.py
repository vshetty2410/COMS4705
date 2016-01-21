import providedcode
from providedcode.transitionparser import TransitionParser
from providedcode.evaluate import DependencyEvaluator
from providedcode.dependencygraph import DependencyGraph
import fileinput
import sys

if not sys.argv[1]:
    print "Model required. Needs to be first argument."
    sys.exit()

tp = TransitionParser.load(sys.argv[1])

for line in sys.stdin:
    sentence = DependencyGraph.from_sentence(line)
    parsed = tp.parse([sentence])
    print parsed[0].to_conll(10).encode('utf-8')