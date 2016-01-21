class Transition(object):
    """
    This class defines a set of transitions which are applied to a
    configuration to get the next configuration.
    """
    # Define set of transitions
    LEFT_ARC = 'LEFTARC'
    RIGHT_ARC = 'RIGHTARC'
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'

    def __init__(self):
        raise ValueError('Do not construct this object!')

    @staticmethod
    def left_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1

        # first, verify that the stack has more than just the root
        if len(conf.stack) == 1:
            return -1

        # next, verify that A doesn't already have the element from the top of the stack
        idx_wi = conf.stack[-1]
        idx_wj = conf.buffer[0]

        for arc in conf.arcs:
            if arc[2] == idx_wi:
                return -1

        # since it's valid, we can pop off the end of the stack
        conf.stack.pop()

        # finally, append to arcs
        conf.arcs.append((idx_wj, relation, idx_wi))

    @staticmethod
    def right_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1

        # You get this one for free! Use it as an example.

        idx_wi = conf.stack[-1]
        idx_wj = conf.buffer.pop(0)

        conf.stack.append(idx_wj)
        conf.arcs.append((idx_wi, relation, idx_wj))

    @staticmethod
    def reduce(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.stack:
            return -1

        # precondition: last element of stack needs to already be in arcs
        idx_wi = conf.stack[-1]

        exists = False
        for arc in conf.arcs:
            if arc[2] == idx_wi:
                exists = True
            
        if not exists:
            return -1

        # finally, we can pop off the end of the stack
        conf.stack.pop()


    @staticmethod
    def shift(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer:
            return -1

        idx_wj = conf.buffer.pop(0)
        conf.stack.append(idx_wj)