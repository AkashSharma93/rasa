from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections import ChainMap
from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, List, Optional, Text, Type

import rasa.shared.utils.common
from rasa.engine.model_storage import ModelStorage, Resource

logger = logging.getLogger(__name__)


@dataclass
class SchemaNode:
    """Represents one node in the schema."""

    needs: Dict[Text, Text]
    uses: Type[GraphComponent]
    constructor_name: Text
    fn: Text
    config: Dict[Text, Any]
    eager: bool = False
    is_target: bool = False
    is_input: bool = False
    resource_name: Optional[Text] = None


GraphSchema = Dict[Text, SchemaNode]


class GraphComponent(ABC):
    """Interface for any component which will run in a graph."""

    # TODO: This doesn't enforce that it exists in subclasses..
    default_config: Dict[Text, Any]

    @classmethod
    @abstractmethod
    def create(
        cls,
        config: Dict,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Creates a new graph component.

        Args:
            config: This config overrides the `default_config`
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            execution_context: Information about the current graph run.

        Returns: An instantiated GraphComponent
        """
        ...

    @classmethod
    def load(
        cls,
        config: Dict,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> GraphComponent:
        """The load method is for creating a component using persisted data.

        Args:
            config: This config overrides the `default_config`
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            execution_context: Information about the current graph run.
            kwargs: Output values from previous nodes might be passed in as `kwargs`.

        Args:
            config: This config overrides the `default_config`
            execution_context: Information about the current graph run.

        Returns: An instantiated, loaded GraphComponent
        """
        return cls.create(config, model_storage, resource, execution_context)

    @abstractmethod
    def supported_languages(self) -> List[Text]:
        """Determines which languages this component can work with."""
        ...

    @abstractmethod
    def required_packages(self) -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        ...


@dataclass
class ExecutionContext:
    """Holds information about a single graph run."""

    graph_schema: GraphSchema = field(repr=False)
    model_id: Text
    diagnostic_data: bool = False
    is_finetuning: bool = False


class GraphNode:
    """Instantiates and runs a `GraphComponent` within a graph.

    A `GraphNode` is a wrapper for a `GraphComponent` that allows it to be executed
    In the context of a graph. It is responsible for instantiating the component at the
    correct time, collecting the inputs from the parent nodes, running the run function
    of the component and passing the output onwards.
    """

    def __init__(
        self,
        node_name: Text,
        component_class: Type[GraphComponent],
        constructor_name: Text,
        component_config: Dict[Text, Any],
        fn_name: Text,
        inputs: Dict[Text, Text],
        eager: bool,
        model_storage: ModelStorage,
        resource_name: Optional[Text],
        execution_context: ExecutionContext,
    ) -> None:
        """Initializes `GraphNode`.

        Args:
            node_name: The name of the node in the schema.
            component_class: The class to be instantiated and run.
            constructor_name: The method used to instantiate the component.
            component_config: Config to be passed to the component.
            fn_name: The function to be run when the node executes.
            inputs: A map from input name to parent node name that provides it.
            eager: Determines if the node is instantiated right away, or just before
                being run.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource_name: If given the `GraphComponent` will be loaded from the
                `model_storage` using the given resource.
            execution_context: Information about the current graph run.
        """
        self._node_name: Text = node_name
        self._component_class: Type[GraphComponent] = component_class
        self._constructor_name: Text = constructor_name
        self._constructor_fn: Callable = getattr(
            self._component_class, self._constructor_name
        )
        self._component_config: Dict[Text, Any] = {
            **self._component_class.default_config,
            **component_config,
        }
        self._fn_name: Text = fn_name
        self._fn: Callable = getattr(self._component_class, self._fn_name)
        self._inputs: Dict[Text, Text] = inputs
        self._eager: bool = eager

        self._model_storage = model_storage
        self._existing_resource_name = resource_name

        self._execution_context: ExecutionContext = execution_context

        self._component: Optional[GraphComponent] = None
        if self._eager:
            self._load_component()

    def _load_component(
        self, additional_kwargs: Optional[Dict[Text, Any]] = None
    ) -> None:
        kwargs = additional_kwargs if additional_kwargs else {}

        logger.debug(
            f"Node {self._node_name} loading "
            f"{self._component_class.__name__}.{self._constructor_name} "
            f"with config: {self._component_config}, and kwargs: {kwargs}."
        )

        self._component: GraphComponent = getattr(  # type: ignore[no-redef]
            self._component_class, self._constructor_name
        )(
            self._component_config,
            self._model_storage,
            self._get_resource(kwargs),
            execution_context=self._execution_context,
            **kwargs,
        )

    def _get_resource(self, kwargs: Dict[Text, Any]) -> Resource:
        if "resource" in kwargs:
            # A parent node provides resource during training. The component wrapped
            # by this `GraphNode` will load itself from this resource.
            return kwargs.pop("resource")
        if self._existing_resource_name:
            # The component should be loaded from a trained resource during inference.
            # E.g. a classifier might train and persist itself during training and will
            # then load itself from this resource during inference.
            return Resource(self._existing_resource_name)
        else:
            # The component gets a chance to persist itself
            return Resource(self._node_name)

    def parent_node_names(self) -> List[Text]:
        """The names of the parent nodes of this node."""
        return list(self._inputs.values())

    def __call__(self, *inputs_from_previous_nodes: List[Any]) -> Dict[Text, Any]:
        """This method is called when the node executes in the graph."""
        received_inputs = dict(ChainMap(*inputs_from_previous_nodes))
        kwargs = {}
        for input, input_node in self._inputs.items():
            kwargs[input] = received_inputs[input_node]

        if not self._component:
            constructor_kwargs = rasa.shared.utils.common.minimal_kwargs(
                kwargs, self._constructor_fn
            )
            self._load_component(constructor_kwargs)

        run_kwargs = rasa.shared.utils.common.minimal_kwargs(kwargs, self._fn)
        logger.debug(
            f"Node {self._node_name} running "
            f"{self._component_class.__name__}.{self._fn_name} "
            f"with kwargs: {run_kwargs}"
        )
        return {self._node_name: self._fn(self._component, **run_kwargs)}

    @classmethod
    def from_schema_node(
        cls,
        node_name: Text,
        schema_node: SchemaNode,
        model_storage: ModelStorage,
        execution_context: ExecutionContext,
    ) -> GraphNode:
        """Creates a `GraphNode` from a `SchemaNode`."""
        return cls(
            node_name=node_name,
            component_class=schema_node.uses,
            constructor_name=schema_node.constructor_name,
            component_config=schema_node.config,
            fn_name=schema_node.fn,
            inputs=schema_node.needs,
            eager=schema_node.eager,
            model_storage=model_storage,
            execution_context=execution_context,
            resource_name=schema_node.resource_name,
        )


def graph_schema_as_dict(graph_schema: GraphSchema) -> Dict[Text, Any]:
    """Returns graph schema in a serializable format.

    Args:
        graph_schema: The graph schema to serialize.

    Returns:
        The graph schema in a format which can be dumped as JSON or other formats.
    """
    serializable_graph_schema = {}
    for node_name, node in graph_schema.items():
        serializable = dataclasses.asdict(node)

        # Classes are not JSON serializable (surprise)
        serializable["uses"] = f"{node.uses.__module__}.{node.uses.__name__}"

        serializable_graph_schema[node_name] = serializable

    return serializable_graph_schema


def load_graph_schema(serialized_graph_schema: Dict[Text, Any]) -> GraphSchema:
    """Loads a graph schema which has been serialized using `graph_schema_as_dict`.

    Args:
        serialized_graph_schema: A serialized graph schema.

    Returns:
        A properly loaded schema.
    """
    graph_schema = {}
    for node_name, serialized_node in serialized_graph_schema.items():
        # TODO: Test and handle error here
        serialized_node["uses"] = rasa.shared.utils.common.class_from_module_path(
            serialized_node["uses"]
        )
        graph_schema[node_name] = SchemaNode(**serialized_node)

    return graph_schema
