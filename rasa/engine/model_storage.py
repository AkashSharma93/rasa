from __future__ import annotations
import logging
import shutil
import tarfile
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Text, ContextManager, TypeVar, Dict, Any, Tuple, Union, TypedDict

import rasa.shared.utils.io
from rasa.shared.core.domain import Domain

logger = logging.getLogger(__name__)

# TODO: Reference existing one
GraphSchema = TypeVar("GraphSchema")

# Paths within model archive
MODEL_ARCHIVE_COMPONENTS_DIR = "components"
MODEL_ARCHIVE_TRAIN_SCHEMA_FILE = "train_schema.yml"
MODEL_ARCHIVE_PREDICT_SCHEMA_FILE = "predict_schema.yml"
MODEL_ARCHIVE_METADATA_FILE = "metadata.json"

# Keys for metadata entries
METADATA_TRAINED_AT_KEY = "trained_at"
METADATA_RASA_VERSION_KEY = "rasa_open_source_version"
METADATA_MODEL_ID_KEY = "model_id"
METADATA_DOMAIN_KEY = "domain"


class ModelMetadata(TypedDict, total=False):
    """Describes a trained model."""

    METADATA_TRAINED_AT_KEY: float
    METADATA_RASA_VERSION_KEY: Text
    METADATA_MODEL_ID_KEY: Text


@dataclass
class Resource:
    """Represents a persisted graph component in the graph."""

    name: Text

    @classmethod
    def from_cache(
        cls, cache_directory: Path, resource_name: Text, model_storage: ModelStorage
    ) -> Resource:
        """Loads a `Resource` from the cache.

        This automatically loads the persisted resource into the given `ModelStorage`.

        Args:
            cache_directory: The directory with the cached `Resource`.
            resource_name: The name of the `Resource`.
            model_storage: The `ModelStorage` which the cached `Resource` will be added
                to so that the `Resource` is accessible for other graph nodes.

        Returns:
            The ready-to-use and accessible `Resource`.
        """
        logger.debug(f"Loading resource '{resource_name}' from cache.")

        resource = Resource(resource_name)
        with model_storage.write_to(resource) as resource_directory:
            shutil.copytree(cache_directory, resource_directory, dirs_exist_ok=True)

        logger.debug(f"Successfully initialized resource '{resource_name}' from cache.")

        return resource

    def to_cache(self, cache_directory: Path, model_storage: ModelStorage) -> None:
        """Persists the `Resource` to the cache.

        Args:
            cache_directory: The directory which receives the persisted `Resource`.
            model_storage: The model storage which currently contains the persisted
                `Resource`
        """
        with model_storage.read_from(self) as resource_directory:
            shutil.copytree(resource_directory, cache_directory, dirs_exist_ok=True)

    def fingerprint(self) -> Text:
        """Provides fingerprint for `Resource`.

        The fingerprint can be just the name as the persisted resource only changes
        if the used training data (which is loaded in previous nodes) or the config
        (which is fingerprinted separately) changes.

        Returns:
            Fingerprint for `Resource`.
        """
        return self.name


class ModelStorage:
    """Stores and provides output of `GraphComponents` which persist themselves."""

    def __init__(self, storage_path: Union[Text, Path]) -> None:
        """Creates the storage.

        Args:
            storage_path: Directory which will contain the persisted graph components.
        """
        self._storage_path = Path(storage_path)

    @contextmanager
    def write_to(self, resource: Resource) -> ContextManager[Path]:
        """Persists data for a given resource.

        This `Resource` can then be accessed in dependent graph nodes via
        `model_storage.read_from`.

        Args:
            resource: The resource which should be persisted.

        Returns:
            A directory which can be used to persist data for the given `Resource`.
        """
        logger.debug(f"Resource '{resource.name}' was requested for writing.")
        directory = self._directory_for_resource(resource)

        if not directory.exists():
            directory.mkdir()

        yield directory

        logger.debug(f"Resource '{resource.name}' was persisted.")

    def _directory_for_resource(self, resource: Resource) -> Path:
        return self._storage_path / resource.name

    @contextmanager
    def read_from(self, resource: Resource) -> ContextManager[Path]:
        """Provides the data of a persisted `Resource`.

        Args:
            resource: The `Resource` whose persisted should be accessed.

        Returns:
            A directory containing the data of the persisted `Resource`.

        Raises:
            ValueError: In case no persisted data for the given `Resource` exists.
        """
        logger.debug(f"Resource '{resource.name}' was requested for reading.")
        directory = self._directory_for_resource(resource)

        if not directory.exists():
            raise ValueError(
                f"Resource '{resource.name}' does not exist. Please make "
                f"sure that the graph component providing the resource "
                f"is a parent node of the current graph node "
                f"(in case this happens during training) or that the "
                f"resource was actually persisted during training "
                f"(in case this happens during inference)."
            )

        yield directory

    def create_model_package(
        self,
        model_archive_path: Union[Text, Path],
        train_schema: GraphSchema,
        predict_schema: GraphSchema,
        domain: Domain,
        model_metadata: Dict[Text, Any],
    ) -> None:
        """Creates a model archive containing all data to load and run the model.

        Args:
            model_archive_path: The path to the archive which should be created.
            train_schema: The schema which was used to train the graph model.
            predict_schema: The schema for running predictions with the trained model.
            domain: The `Domain` which was used to train the model.
            model_metadata: Any additional metadata which should be stored with the
                model.
        """
        logger.debug(f"Start to created model package for path '{model_archive_path}'.")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            shutil.copytree(self._storage_path, temp_dir / MODEL_ARCHIVE_COMPONENTS_DIR)

            self._persist_schemas(train_schema, predict_schema, temp_dir)
            self._persist_metadata(domain, model_metadata, temp_dir)

            with tarfile.open(model_archive_path, "w:gz") as tar:
                tar.add(temp_dir, arcname="")

        logger.debug(f"Model package created in path '{model_archive_path}'.")

    @staticmethod
    def _persist_schemas(
        train_schema: GraphSchema,
        predict_schema: GraphSchema,
        temporary_directory: Path,
    ) -> None:
        for filename, schema in zip(
            [MODEL_ARCHIVE_TRAIN_SCHEMA_FILE, MODEL_ARCHIVE_PREDICT_SCHEMA_FILE],
            [train_schema, predict_schema],
        ):
            rasa.shared.utils.io.write_yaml(schema, temporary_directory / filename)

    @staticmethod
    def _persist_metadata(
        domain: Domain, model_metadata: Dict[Text, Any], temporary_directory: Path
    ) -> None:
        model_metadata = model_metadata.copy()
        model_metadata[METADATA_DOMAIN_KEY] = domain.as_dict()
        model_metadata[METADATA_TRAINED_AT_KEY] = time.time()
        model_metadata[METADATA_MODEL_ID_KEY] = uuid.uuid4().hex
        model_metadata[METADATA_RASA_VERSION_KEY] = rasa.__version__

        rasa.shared.utils.io.dump_obj_as_json_to_file(
            temporary_directory / MODEL_ARCHIVE_METADATA_FILE, model_metadata
        )

    def unpack(
        self, model_archive_path: Union[Text, Path]
    ) -> Tuple[GraphSchema, GraphSchema, Domain, ModelMetadata]:
        """Unpacks a model archive and initializes `ModelStorage`.

        Args:
            model_archive_path: The path to the model archive.

        Returns:
            Train graph schema, predict graph schema, the domain used for the training
            and the model metadata.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_directory = Path(temporary_directory)

            self._extract_archive_to_directory(model_archive_path, temporary_directory)
            logger.debug(f"Extracted model to '{temporary_directory}'.")

            self._initialize_model_storage_from_model_archive(temporary_directory)

            train_schema, predict_schema, = self._read_schemas(temporary_directory)
            metadata = self._load_metadata(temporary_directory)
            domain = self._extract_domain_from_metadata(metadata)

            return (
                train_schema,
                predict_schema,
                domain,
                metadata,
            )

    @staticmethod
    def _extract_archive_to_directory(
        model_archive_path: Union[Text, Path], temporary_directory: Union[Text, Path],
    ) -> None:
        with tarfile.open(model_archive_path, mode="r:gz") as tar:
            tar.extractall(temporary_directory)

    def _initialize_model_storage_from_model_archive(
        self, temporary_directory: Path
    ) -> None:
        for path in (temporary_directory / MODEL_ARCHIVE_COMPONENTS_DIR).glob("*"):
            shutil.move(
                str(path), str(self._storage_path),
            )

    @staticmethod
    def _extract_domain_from_metadata(metadata: Dict[Text, Any]) -> Domain:
        serialized_domain = metadata.pop(METADATA_DOMAIN_KEY)
        return Domain.from_dict(serialized_domain)

    @staticmethod
    def _load_metadata(directory: Path) -> Dict[Text, Any]:
        return rasa.shared.utils.io.read_json_file(
            directory / MODEL_ARCHIVE_METADATA_FILE
        )

    @staticmethod
    def _read_schemas(directory: Path) -> Tuple[GraphSchema, GraphSchema]:
        return (
            rasa.shared.utils.io.read_yaml_file(
                directory / MODEL_ARCHIVE_TRAIN_SCHEMA_FILE
            ),
            rasa.shared.utils.io.read_yaml_file(
                directory / MODEL_ARCHIVE_PREDICT_SCHEMA_FILE
            ),
        )
