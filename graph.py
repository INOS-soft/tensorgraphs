import networkx
import functools
import ops

FORWARD_MODE = 'fwd'
BACKWARD_MODE = 'back'

class NoSourceException(Exception):
    pass

class TensorGraph(object):
    def __init__(self):
        self.source_nodes = []
        self.terminal_nodes = []
        self._graph_node_map = {}
        self.source_values = {}
        self.graph = networkx.Graph()

    def add_source_node(self, name):
        source_node = SourceGraphNode(name, self)
        self.graph.add_node(source_node)
        self.source_nodes.append(source_node)
        return source_node

    def get_edge(self, n1, n2):
        return self.graph.get_edge_data(n1, n2)

    def add_terminal_node(self, final_node, name):
        terminal_node = TerminalGraphNode(final_node, name, self)
        self.terminal_nodes.append(terminal_node)
        self.graph.add_node(terminal_node)
        self.graph.add_edge(final_node, terminal_node, mode=FORWARD_MODE)
        self.graph.add_edge(terminal_node, final_node, mode=BACKWARD_MODE, arg_num=0)
        final_node.output_nodes.append(terminal_node)
        return terminal_node

    def add_op(self, op, input_nodes, **kwargs):
        if not isinstance(input_nodes, list):
            input_nodes = [input_nodes]
        node = OpGraphNode(input_nodes, op, self, **kwargs)
        self.graph.add_node(node, name=op.name)
        for i, input_node in enumerate(input_nodes):
            self.graph.add_edge(input_node, node, mode=FORWARD_MODE, arg_num=i)
            self.graph.add_edge(node, input_node, mode=BACKWARD_MODE, arg_num=i)
            input_node.output_nodes.append(node)
        return node

    def evaluate_graph(self, feed_dict, terminal_nodes=None):
        self.source_values = feed_dict
        terminal_nodes = terminal_nodes or self.terminal_nodes

        if not isinstance(terminal_nodes, list):
            return terminal_nodes.fwd()
        else:
            return [(n.name, n.fwd()) for n in self.terminal_nodes]

    def get_source_value(self, name):
        try:
            return self.source_values[name]
        except KeyError:
            raise NoSourceException('No value provided for source node: {}'.format(name))

    def reset(self):
        for n in self.graph.nodes:
            n.output = None
            n.grads_self = None
            n.grads_input = None

class TensorGraphNode(object):

    def __init__(self, graph, name):
        self.name = name
        self._graph = graph
        self.output = None
        self.grads_self = None
        self.grads_input = None
        self.input_nodes = []
        self.output_nodes = []

    def fwd(self):
        if self.output is None:
            args = [node.fwd() for node in self.input_nodes]
            self.output = self.fwd_fn(*args)
            self.grads_self = None
            self.grads_input = None

        assert self.output is not None
        return self.output

    def _back_inner(self, nodes_to_search):

        if self.grads_input is not None:
            return self.grads_input

        inputs = [node.fwd() for node in self.input_nodes]
        output_grads = []
        for node in self.output_nodes:
            if node in nodes_to_search:
                arg_num = self._graph.get_edge(node, self)['arg_num']
                output_grad = node._back_inner(nodes_to_search)[arg_num]
                output_grads.append(output_grad)

        grad = self.back_fn(inputs, sum(output_grads))
        if self.grads_self is not None:
            self.grads_self += grad[0]
        else:
            self.grads_self = grad[0]

        if self.grads_input is not None:
            self.grads_input = [g1 + g2 for (g1, g2) in zip(self.grads_input, grad[1])]
        else:
            self.grads_input = grad[1]

        return grad[1]

    def back(self, to):
        nodes_to_search = set()
        for path in networkx.all_simple_paths(self._graph.graph, source=self, target=to):
            nodes_to_search.update(set(path))

        return [node._back_inner(nodes_to_search) for node in to]




class OpGraphNode(TensorGraphNode):
    def __init__(self, input_nodes, op, graph, **kwargs):
        super().__init__(graph, name=op.name)
        self.input_nodes = input_nodes

        self.fwd_fn = functools.partial(op.__call__, **kwargs)
        self.back_fn = op.grad


class SourceGraphNode(TensorGraphNode):
    def __init__(self, name, graph):
        super().__init__(graph, name=name)
        self.fwd_fn = lambda: self._graph.get_source_value(self.name)
        self.back_fn = lambda u, v: ([], v)

    def fwd(self):
        return self.fwd_fn()


class TerminalGraphNode(TensorGraphNode):
    def __init__(self, node, name, graph):
        super().__init__(graph, name=name)
        self.fwd_fn = lambda x: x
        self.back_fn = lambda u, v: ([], [1.0])
        self.input_nodes = [node]

    def eval(self, feed_dict):
        return self._graph.evaluate_graph(feed_dict=feed_dict, terminal_nodes=self)