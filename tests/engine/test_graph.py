from __future__ import annotations

from pathlib import Path
from typing import Dict, Text, Any, List

import rasa.shared.utils.io
from rasa.engine import graph
from rasa.engine.graph import GraphNode, GraphComponent, ExecutionContext, SchemaNode
from rasa.engine.model_storage import ModelStorage, Resource


class PersistableTestComponent(GraphComponent):
    default_config = {}

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        eager_instantiated_value: Any = None,
    ) -> None:
        self._model_storage = model_storage
        self._resource = resource
        self._config = config
        self._eager_instantiated_value = eager_instantiated_value

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> PersistableTestComponent:
        assert model_storage
        assert resource

        return cls(config, model_storage, resource)

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> PersistableTestComponent:
        assert model_storage
        assert resource

        with model_storage.read_from(resource) as directory:
            eager_instantiated_value = rasa.shared.utils.io.read_json_file(
                directory / "test.json"
            )
        return cls(config, model_storage, resource, eager_instantiated_value)

    def supported_languages(self) -> List[Text]:
        return []

    def required_packages(self) -> List[Text]:
        return []

    def train(self) -> Resource:
        with self._model_storage.write_to(self._resource) as directory:
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                directory / "test.json", self._config["test_value"]
            )
            sub_dir = directory / "sub_dir"
            sub_dir.mkdir()

            rasa.shared.utils.io.dump_obj_as_json_to_file(
                sub_dir / "test.json", self._config["test_value_for_sub_directory"]
            )

        return self._resource

    def run_train_process(self) -> Any:
        return self._eager_instantiated_value

    def run_inference(self) -> Any:
        return self._eager_instantiated_value


def test_writing_to_resource_during_training(default_model_storage: ModelStorage):
    node_name = "some_name"

    test_value_for_sub_directory = {"test": "test value"}
    test_value = {"test sub dir": "test value sub dir"}

    node = GraphNode(
        node_name=node_name,
        component_class=PersistableTestComponent,
        constructor_name="create",
        component_config={
            "test_value": test_value,
            "test_value_for_sub_directory": test_value_for_sub_directory,
        },
        fn_name="train",
        inputs={},
        eager=False,
        model_storage=default_model_storage,
        resource_name=None,
        execution_context=ExecutionContext({}, "123"),
    )

    resource = node()[node_name]

    assert resource == Resource(node_name)

    with default_model_storage.read_from(resource) as directory:
        assert (
            rasa.shared.utils.io.read_json_file(directory / "test.json") == test_value
        )
        assert (
            rasa.shared.utils.io.read_json_file(directory / "sub_dir" / "test.json")
            == test_value_for_sub_directory
        )


def test_loading_from_resource_not_eager(default_model_storage: ModelStorage):
    previous_resource = Resource("previous resource")
    parent_node_name = "parent"
    test_value = {"test": "test value"}

    # Pretend resource persisted itself before
    with default_model_storage.write_to(previous_resource) as directory:
        rasa.shared.utils.io.dump_obj_as_json_to_file(
            directory / "test.json", test_value
        )

    node_name = "some_name"
    node = GraphNode(
        node_name=node_name,
        component_class=PersistableTestComponent,
        constructor_name="load",
        component_config={},
        fn_name="run_train_process",
        inputs={"resource": parent_node_name},
        eager=False,
        model_storage=default_model_storage,
        resource_name=None,
        execution_context=ExecutionContext({}, "123"),
    )

    value = node({parent_node_name: previous_resource})[node_name]

    assert value == test_value


def test_loading_from_resource_eager(default_model_storage: ModelStorage):
    previous_resource = Resource("previous resource")
    test_value = {"test": "test value"}

    # Pretend resource persisted itself before
    with default_model_storage.write_to(previous_resource) as directory:
        rasa.shared.utils.io.dump_obj_as_json_to_file(
            directory / "test.json", test_value
        )

    node_name = "some_name"
    node = GraphNode(
        node_name=node_name,
        component_class=PersistableTestComponent,
        constructor_name="load",
        component_config={},
        fn_name="run_inference",
        inputs={},
        eager=True,
        model_storage=default_model_storage,
        resource_name=previous_resource.name,
        execution_context=ExecutionContext({}, "123"),
    )

    value = node()[node_name]

    assert value == test_value


def test_serialize_graph_schema(tmp_path: Path):
    graph_schema = {
        "train": SchemaNode(
            needs={},
            uses=PersistableTestComponent,
            fn="train",
            constructor_name="create",
            config={"some_config": 123455, "some more config": [{"nested": "hi"}]},
        ),
        "load": SchemaNode(
            needs={"resource": "train"},
            uses=PersistableTestComponent,
            fn="run_inference",
            constructor_name="load",
            config={},
            is_target=True,
        ),
    }

    serialized = graph.graph_schema_as_dict(graph_schema)

    # Dump it to make sure it's actually serializable
    file_path = tmp_path / "my_graph.yml"
    rasa.shared.utils.io.write_yaml(serialized, file_path)

    serialized_graph_schema_from_file = rasa.shared.utils.io.read_yaml_file(file_path)
    graph_schema_from_file = graph.load_graph_schema(serialized_graph_schema_from_file)

    assert graph_schema_from_file == graph_schema
