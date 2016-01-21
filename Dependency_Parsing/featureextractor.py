from nltk.compat import python_2_unicode_compatible

printed = False

@python_2_unicode_compatible
class FeatureExtractor(object):
    @staticmethod
    def _check_informative(feat, underscore_is_informative=False):
        """
        Check whether a feature is informative
        """

        if feat is None:
            return False

        if feat == '':
            return False

        if not underscore_is_informative and feat == '_':
            return False

        return True

    @staticmethod
    def find_left_right_dependencies(idx, arcs):
        left_most = 1000000
        right_most = -1
        dep_left_most = ''
        dep_right_most = ''
        for (wi, r, wj) in arcs:
            if wi == idx:
                if (wj > wi) and (wj > right_most):
                    right_most = wj
                    dep_right_most = r
                if (wj < wi) and (wj < left_most):
                    left_most = wj
                    dep_left_most = r
        return dep_left_most, dep_right_most

    @staticmethod
    def extract_features(tokens, buffer, stack, arcs):
        """
        This function returns a list of string features for the classifier
        :param tokens: nodes in the dependency graph
        :param stack: partially processed words
        :param buffer: remaining input words
        :param arcs: partially built dependency tree
        :return: list(str)
        """

        """
        Think of some of your own features here! Some standard features are
        described in Table 3.2 on page 31 of Dependency Parsing by Kubler,
        McDonald, and Nivre
        [http://books.google.com/books/about/Dependency_Parsing.html?id=k3iiup7HB9UC]
        'word': word,
        'lemma': '_',
        'ctag': tag,
        'tag': tag,
        'feats': '_',
        'rel': '_',
        'deps': defaultdict(),
        'head': '_',
        'address': index + 1,
        """

        result = []


        global printed
        if not printed:
            #print("This is VERY GOOD feature extractor!")
            printed = True

        # set of features:

        # Features that involve both stack and buffer
        if stack and buffer:
            # distance between STK[0] and BUF[0]
            stack_idx0 = stack[-1]
            stack_token = tokens[stack_idx0]
            
            buffer_idx0 = buffer[0]
            buffer_token = tokens[buffer_idx0]

            result.append('STK_0_BUF_0_DIST_' + str(abs(stack_idx0 - buffer_idx0)))

            # number of VERBs between STK[0] and BUF[0]
            count = 0
            for i in xrange(stack_idx0, buffer_idx0):
                if tokens[i]['ctag'] == 'VERB':
                    count += 1

            result.append('STK_0_BUF_0_NUMVERBS_' + str(count))

        if stack:
            # first, STK[0]:
            # FORM, LEMMA, POSTAG, FEATS
            stack_idx0 = stack[-1]
            token = tokens[stack_idx0]
            if FeatureExtractor._check_informative(token['word'], True):
                result.append('STK_0_FORM_' + token['word'])

            if FeatureExtractor._check_informative(token['lemma']):
                result.append('STK_0_LEMMA_' + token['lemma'])

            if FeatureExtractor._check_informative(token['tag']):
                result.append('STK_0_TAG_' + token['tag'])

            if 'feats' in token and FeatureExtractor._check_informative(token['feats']):
                feats = token['feats'].split("|")
                for feat in feats:
                    result.append('STK_0_FEATS_' + feat)

            # next, STK[1]:
            # POSTAG
            if len(stack) >= 2:
                stack_idx1 = stack[-2]
                token = tokens[stack_idx1]
                if FeatureExtractor._check_informative(token['tag']):
                    result.append('STK_1_TAG_' + token['tag'])

            # next, LDEP and RDEP of STK[0]:
            # Left most, right most dependency of stack[0]
            dep_left_most, dep_right_most = FeatureExtractor.find_left_right_dependencies(stack_idx0, arcs)

            if FeatureExtractor._check_informative(dep_left_most):
                result.append('STK_0_LDEP_' + dep_left_most)
            if FeatureExtractor._check_informative(dep_right_most):
                result.append('STK_0_RDEP_' + dep_right_most)

            # next, left children and right children of STK[0]
            lchildren = 0
            rchildren = 0
            token = stack[-1]
            for arc in arcs:
                if tokens[arc[0]] == token:
                    if tokens[arc[2]]['address'] > token['address']:
                        rchildren += 1
                    else:
                        lchildren += 1

            result.append('STK_0_LCHILD_' + str(lchildren))
            result.append('STK_0_RCHILD_' + str(rchildren))

            

        if buffer:
            # first, BUF[0]:
            # FORM, LEMMA, POSTAG, FEATS
            buffer_idx0 = buffer[0]
            token = tokens[buffer_idx0]
            if FeatureExtractor._check_informative(token['word'], True):
                result.append('BUF_0_FORM_' + token['word'])

            if FeatureExtractor._check_informative(token['lemma']):
                result.append('BUF_0_LEMMA_' + token['lemma'])

            if FeatureExtractor._check_informative(token['tag']):
                result.append('BUF_0_TAG_' + token['tag'])

            if 'feats' in token and FeatureExtractor._check_informative(token['feats']):
                feats = token['feats'].split("|")
                for feat in feats:
                    result.append('BUF_0_FEATS_' + feat)

            # next, BUF[1]:
            # FORM, POSTAG
            if len(buffer) >= 2:
                buffer_idx1 = buffer[1]
                token = tokens[buffer_idx1]
                if FeatureExtractor._check_informative(token['word'], True):
                    result.append('BUF_1_FORM_' + token['word'])

                if FeatureExtractor._check_informative(token['tag']):
                    result.append('BUF_1_TAG_' + token['tag'])


            # next, BUF[2]:
            # POSTAG
            if len(buffer) >= 3:
                buffer_idx2 = buffer[2]

                if FeatureExtractor._check_informative(token['tag']):
                    result.append('BUF_2_TAG_' + token['tag'])

            # next, BUF[3]:
            # POSTAG
            if len(buffer) >= 4:
                buffer_idx3 = buffer[3]

                if FeatureExtractor._check_informative(token['tag']):
                    result.append('BUF_3_TAG_' + token['tag'])

            # next, LDEP and RDEP of BUF[0]:
            dep_left_most, dep_right_most = FeatureExtractor.find_left_right_dependencies(buffer_idx0, arcs)

            if FeatureExtractor._check_informative(dep_left_most):
                result.append('BUF_0_LDEP_' + dep_left_most)
            if FeatureExtractor._check_informative(dep_right_most):
                result.append('BUF_0_RDEP_' + dep_right_most)

            # next, left children and right children of BUF[0]
            lchildren = 0
            rchildren = 0
            token = buffer[0]
            for arc in arcs:
                if tokens[arc[0]] == token:
                    if tokens[arc[2]]['address'] > token['address']:
                        rchildren += 1
                    else:
                        lchildren += 1

            result.append('BUF_0_LCHILD_' + str(lchildren))
            result.append('BUF_0_RCHILD_' + str(rchildren))

        return result