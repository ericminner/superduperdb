import dataclasses as dc
import typing as t

import networkx as nx

from superduperdb import Schema
from superduperdb.backends.base.query import Select
from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.components.model import Signature, _Predictor


def input_node(*args):
    return IndexableNode(
        model=Input(spec=args if len(args) > 1 else args[0]),
        parent_graph=nx.DiGraph(),
        parent_models={}
    )


def document_node(*args):
    return IndexableNode(
        model=DocumentInput(spec=args),
        parent_graph=nx.DiGraph(),
        parent_models={}
    )


class IndexableNode:
    def __init__(self, *, model, parent_graph, parent_models, index=None,
                 identifier=None):
        self.model = model
        self.parent_graph = parent_graph
        self.parent_models = parent_models
        self.index = index
        self.identifier = identifier
    
    def __getitem__(self, item):
        return IndexableNode(
            model=self.model,
            parent_graph=self.parent_graph,
            parent_models=self.parent_models,
            index=item,
            identifier=self.identifier,
        )

    def to_graph(self, identifier: str):
        from superduperdb.components.graph import Graph, DocumentInput
        input_model = next(
            v for k, v in self.parent_models.items()
            if isinstance(self.parent_models[k].model, (Input, DocumentInput))
        )
        graph = Graph(
            identifier=identifier,
            input=input_model.model,
            outputs=[self.model]
        )
        for u, v, data in self.parent_graph.edges(data=True):
            u_node = self._get_node(u)
            v_node = self._get_node(v)
            graph.connect(u_node.model, v_node.model, on=(u_node.index, data['key']))
        return graph

    def _get_node(self, u):
        if u == self.model.identifier:
            return self
        return self.parent_models[u]

    def to_listeners(self, select: Select, identifier: str):
        from superduperdb.components.listener import Listener
        from superduperdb.components.stack import Stack
        nodes = list(nx.topological_sort(self.parent_graph))
        input_node = next(
            v for k, v in self.parent_models.items()
            if isinstance(self.parent_models[k].model, DocumentInput)
        )
        assert isinstance(input_node.model, DocumentInput)
        listener_lookup = {}
        for node in nodes:
            in_edges = list(self.parent_graph.in_edges(node, data=True))
            if not in_edges:
                continue
            node = self._get_node(node)
            key = {}
            for u, _, data in in_edges:
                previous_node = self._get_node(u)
                if isinstance(previous_node.model, DocumentInput):
                    key[previous_node.index] = data['key']
                else:
                    assert previous_node.index is None
                    upstream_listener = listener_lookup[
                        previous_node.model.identifier
                    ]
                    key[upstream_listener.outputs] = data['key']
            listener_lookup[node.model.identifier] = Listener(
                model=node.model,
                select=select,
                key=key,
                identifier=node.identifier,
            )
        return Stack(
            identifier=identifier,
            components=list(listener_lookup.values()),
        )


class OutputWrapper:
    def __init__(self, r, keys):
        self.keys = keys
        self.r = r

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.r[self.keys[item]]
        elif isinstance(item, str):
            return self.r[item]
        else:
            raise TypeError(f'Unsupported type for __getitem__: {type(item)}')


@dc.dataclass(kw_only=True)
class Input(_Predictor):
    spec: t.Union[str, t.List[str]]
    identifier: str = '_input'
    signature: Signature = '*args' 
    def __post_init__(self, artifacts):
        super().__post_init__(artifacts)
        if isinstance(self.spec, str):
            self.signature = 'singleton'

    def predict_one(self, *args):
        if self.signature == 'singleton':
            return args[0]
        return OutputWrapper({k: arg for k, arg in zip(self.spec, args)}, keys=self.spec)

    def predict(self, dataset):
        return [self.predict_one(dataset[i]) for i in range(len(dataset))]


@dc.dataclass(kw_only=True)
class DocumentInput(_Predictor):
    spec: t.Union[str, t.List[str]]
    identifier: str = '_input'
    signature: t.ClassVar[Signature] = 'singleton' 

    def __post_init__(self, artifacts):
        super().__post_init__(artifacts)

    def predict_one(self, r):
        return {k: r[k] for k in self.spec}

    def predict(self, dataset):
        return [self.predict_one(dataset[i]) for i in range(len(dataset))]


@dc.dataclass(kw_only=True)
class Graph(_Predictor):
    '''
    Represents a directed acyclic graph composed of interconnected model nodes.

    This class enables the creation of complex predictive models
    by defining a computational graph structure where each node
    represents a predictive model. Models can be connected via edges
    to define dependencies and flow of data.

    The predict() method executes predictions through the graph, ensuring
    correct data flow and handling of dependencies

    Example:
    >>  g = Graph(
    >>    identifier='simple-graph', input=model1, outputs=[model2], signature='*args'
    >>  )
    >>  g.connect(model1, model2)
    >>  assert g.predict_one(1) == [(4, 2)]

    '''

    models: t.List[_Predictor] = dc.field(default_factory=list)
    edges: t.List[t.Tuple[str, str, t.Tuple[t.Union[int, str], str]]] = dc.field(
        default_factory=list
    )
    input: _Predictor
    outputs: t.List[t.Union[str, _Predictor]] = dc.field(default_factory=list)
    _DEFAULT_ARG_WEIGHT: t.ClassVar[tuple] = (None, 'singleton')
    type_id: t.ClassVar[str] = 'model'

    def __post_init__(self, artifacts):
        self.G = nx.DiGraph()
        self.nodes = {}
        self.version = 0
        self._db = None
        self.signature = self.input.signature

        assert all([isinstance(o, _Predictor) for o in self.outputs])

        self.output_schema = Schema(
            identifier=self.identifier,
            fields={k.identifier: k.datatype for k in self.outputs},
        )

        self.outputs = [
            o.identifier if isinstance(o, _Predictor) else o for o in self.outputs
        ]

        # Load the models and edges into a `DiGraph`
        models = {m.identifier: m for m in self.models}
        if self.edges and models:
            for connection in self.edges:
                u, v, on = connection
                self.connect(
                    models[u],
                    models[v],
                    on=on,
                    update_edge=False,
                )
        super().__post_init__(artifacts=artifacts)

    def connect(
        self,
        u: _Predictor,
        v: _Predictor,
        on: t.Optional[t.Tuple[t.Union[int, str], str]] = None,
        update_edge: t.Optional[bool] = True,
    ):
        '''
        Connects two nodes `u` and `v` on edge where edge is a tuple with
        first element describing outputs index (int or None)
        and second describing input argument (str).

        Note:
        output index: None means all outputs of node u are connected to node v
        '''
        assert isinstance(u, _Predictor)
        assert isinstance(v, _Predictor)

        if u.identifier not in self.nodes:
            if u.identifier != self.input.identifier:
                u.signature = '*args'
            self.nodes[u.identifier] = u
            self.G.add_node(u.identifier)

        if v.identifier not in self.nodes:
            v.signature = '*args'
            self.nodes[v.identifier] = v
            self.G.add_node(v.identifier)

        G_ = self.G.copy()
        G_.add_edge(u.identifier, v.identifier, weight=on or self._DEFAULT_ARG_WEIGHT)
        if not nx.is_directed_acyclic_graph(G_):
            raise TypeError('The graph is not DAG with this edge')
        self.G = G_

        if update_edge:
            self.edges.append(
                (u.identifier, v.identifier, on or self._DEFAULT_ARG_WEIGHT)
            )
            if isinstance(u, _Predictor) and u not in self.models:
                self.models.append(u)
            if v not in self.models:
                self.models.append(v)
        return

    def fetch_outputs(
        self, output, index: t.Optional[t.Union[int, str]] = None,
    ):
        if index is not None:
            assert isinstance(index, (int, str))
            if isinstance(index, str):
                assert isinstance(
                    output, dict
                ), 'Output should be a dict for indexing with str'

            try:
                return [[o[index]] for o in output]
            except KeyError:
                raise KeyError("Model node does not have sufficient outputs")
        else:
            return [[o] for o in output]

    def fetch_output(
        self, output, index: t.Optional[t.Union[int, str]] = None
    ):
        if index is not None:
            assert isinstance(index, (int, str))

            if isinstance(index, str):
                assert isinstance(
                    output, dict
                ), 'Output should be a dict for indexing with str'

            try:
                return output[index]

            except KeyError:
                raise KeyError("Model node does not have sufficient outputs")
        else:
            return output

    def _validate_graph(self, node):
        '''
        Validates the graph for any disconnection
        '''
        # TODO: Create a cache to reduce redundant validation in predict in db

        predecessors = list(self.G.predecessors(node))
        dependencies = [self._validate_graph(node=p) for p in predecessors]
        model = self.nodes[node]
        if dependencies and len(model.inputs) != len(dependencies):
            raise TypeError(
                f'Graph disconnected at Node: {model.identifier} '
                f'and is partially connected with {dependencies}\n'
                f'Required connected node is {len(model.inputs)} '
                f'but got only {len(dependencies)}, '
                f'Node required params: {model.inputs.params}'
            )
        return node

    def _fetch_inputs(self, dataset, edges=[], outputs=[]):
        arg_inputs = []

        def _length(lst):
            count = 0
            for item in lst:
                if isinstance(item, (list, tuple)):
                    count += _length(item)
                else:
                    count += 1
            return count

        if not _length(outputs):
            outputs = dataset

        for ix, edge in enumerate(edges):
            output_key, input_key = edge['weight']

            arg_input_dataset = self.fetch_outputs(outputs[ix], output_key)
            if input_key == 'singleton':
                return arg_input_dataset

            arg_inputs.append(arg_input_dataset)

        if not arg_inputs:
            return outputs
        return self._transpose(arg_inputs, args=True)

    def _fetch_one_inputs(self, args, kwargs, edges=[], outputs=[]):
        node_input = {}
        for ix, edge in enumerate(edges):
            output_key, input_key = edge['weight']
            node_input[input_key] = self.fetch_output(outputs[ix], output_key)
            kwargs = node_input
            args = ()

        if 'singleton' in node_input:
            args = (node_input['singleton'],)
            kwargs = {}

        return args, kwargs

    def _predict_one_on_node(self, *args, node=None, cache={}, **kwargs):
        if node not in cache:
            predecessors = list(self.G.predecessors(node))
            outputs = [
                self._predict_one_on_node(*args, **kwargs, node=p, cache=cache)
                for p in predecessors
            ]
            edges = [self.G.get_edge_data(p, node) for p in predecessors]
            args, kwargs = self._fetch_one_inputs(
                args, kwargs, edges=edges, outputs=outputs
            )
            cache[node] = self.nodes[node].predict_one(*args, **kwargs)
            return cache[node]
        return cache[node]

    def _predict_on_node(self, *args, node=None, cache={}, **kwargs):
        if node not in cache:
            predecessors = list(self.G.predecessors(node))
            outputs = [
                self._predict_on_node(*args, **kwargs, node=p, cache=cache)
                for p in predecessors
            ]
            edges = [self.G.get_edge_data(p, node) for p in predecessors]
            dataset = self._fetch_inputs(args[0], edges=edges, outputs=outputs)
            # we know that dataset is in `**kwargs` format
            # now we can convert it to self.nodes[node].signature format
            cache[node] = self.nodes[node].predict(dataset=dataset)
            return cache[node]
        return cache[node]

    def predict_one(self, *args, **kwargs):
        '''
        Single data point prediction passes the args and kwargs to defined node flow
        in the graph.
        '''
        # Validate the node for incompletion
        list(map(self._validate_graph, self.outputs))
        cache = {}

        outputs = [
            self._predict_one_on_node(*args, node=output, cache=cache, **kwargs)
            for output in self.outputs
        ]

        # TODO: check if output schema and datatype required
        return outputs

    def patch_dataset_to_args(self, dataset):
        '''
        Patch the dataset with args type as default, since all
        corresponding nodes takes args as input type
        '''
        args_dataset = []
        signature = self.signature

        def mapping(x):
            nonlocal signature
            if signature == '**kwargs':
                return list(x.values())
            elif signature == '*args,**kwargs':
                return list(x[0]) + list(x[1].values())
            else:
                return x

        for data in dataset:
            data = mapping(data)
            args_dataset.append(data)
        return args_dataset

    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        # Validate the node for incompletion
        list(map(self._validate_graph, self.outputs))

        if isinstance(dataset, QueryDataset):
            raise TypeError('QueryDataset is not supported in graph mode')
        cache: t.Dict[str, t.Any] = {}

        outputs = [
            self._predict_on_node(dataset, node=output, cache=cache, one=False)
            for output in self.outputs
        ]

        # TODO: check if output schema and datatype required
        return outputs

    def encode_outputs(self, outputs):
        encoded_outputs = []
        for o, n in zip(outputs, self.outputs):
            encoded_outputs.append(self.nodes[n].encode_outputs(o))
        outputs = self._transpose(outputs=encoded_outputs or outputs)
        return self.encode_with_schema(outputs)

    @staticmethod
    def _transpose(outputs, args=False):
        transposed_outputs = []
        for i in range(len(outputs[0])):
            batch_outs = []
            for o in outputs:
                batch_outs.append(o[i][0] if args else o[i])
            transposed_outputs.append(batch_outs)
        return transposed_outputs
