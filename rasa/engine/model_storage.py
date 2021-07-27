import json
import logging
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Text, ContextManager, TypeVar, Dict, Any, Tuple

import fs.move
import fs.copy
import fs.errors
from fs.base import FS
from fs.tarfs import TarFS
import fs.tempfs

import rasa.shared.utils.io
from rasa.shared.core.domain import Domain

logger = logging.getLogger(__name__)

# TODO: Reference existing one
GraphSchema = TypeVar("GraphSchema")

METADATA_DOMAIN_KEY = "domain"
METADATA_TRAINED_AT_KEY = "trained_at"
METADATA_RASA_VERSION_KEY = "rasa_open_source_version"
METADATA_MODEL_ID_KEY = "model_id"

MODEL_ARCHIVE_COMPONENTS_DIR = "components"
MODEL_ARCHIVE_TRAIN_SCHEMA_FILE = "train_schema.yml"
MODEL_ARCHIVE_PREDICT_SCHEMA_FILE = "predict_schema.yml"
MODEL_ARCHIVE_METADATA_FILE = "metadata.json"

# TODO: make dataclass a typed dict


@dataclass
class Resource:

    name: Text

    @classmethod
    def from_cache(
        cls, cache_directory: Path, node_name: Text, model_storage: "ModelStorage"
    ) -> "Resource":
        resource = Resource(node_name)
        with model_storage.write_to(resource) as resource_directory:
            shutil.copytree(cache_directory, resource_directory, dirs_exist_ok=True)

        return resource

    def to_cache(self, cache_directory: Path, model_storage: "ModelStorage") -> None:
        with model_storage.read_from(self) as resource_directory:
            shutil.copytree(resource_directory, cache_directory, dirs_exist_ok=True)


class ModelStorage:
    def __init__(self, storage_path: Text) -> None:
        self._storage_path = storage_path

    @contextmanager
    def write_to(self, resource: Resource) -> ContextManager[Path]:
        with tempfile.TemporaryDirectory() as directory:
            yield Path(directory)

            fs.move.move_fs(directory, self._directory_for_resource(resource))

    def _directory_for_resource(self, resource: Resource) -> Text:
        joined = fs.path.combine(self._storage_path, resource.name)
        return joined

    @contextmanager
    def read_from(self, resource: Resource) -> ContextManager[Path]:
        with tempfile.TemporaryDirectory() as temporary_resource_directory:
            try:
                directory_for_resource = self._directory_for_resource(resource)
                with fs.open_fs(directory_for_resource) as resource_directory:
                    fs.copy.copy_fs(resource_directory, temporary_resource_directory)
            except fs.errors.CreateFailed:
                logger.warning(f"Resource '{resource.name}' does not exist.")

            yield Path(temporary_resource_directory)

    def create_model_package(
        self,
        model_archive_path: Text,
        train_schema: GraphSchema,
        predict_schema: GraphSchema,
        domain: Domain,
        model_metadata: Dict[Text, Any],
    ) -> Text:
        with fs.tempfs.TempFS() as temp_fs:
            model_file = temp_fs.getsyspath("model.tar.gz")

            # move model content to local disk so we can tar it
            with fs.tarfs.TarFS(model_file, write=True) as tar_fs:
                fs.move.move_dir(
                    self._storage_path, "/", tar_fs, MODEL_ARCHIVE_COMPONENTS_DIR,
                )

                self._add_schemas_to_archive(train_schema, predict_schema, tar_fs)
                self._add_metadata_to_archive(domain, model_metadata, tar_fs)

            # copy the model archive back to the provided path
            fs.move.move_file(
                temp_fs,
                "model.tar.gz",
                fs.path.dirname(model_archive_path),
                fs.path.basename(model_archive_path),
            )

        return model_archive_path

    @staticmethod
    def _add_schemas_to_archive(
        train_schema: GraphSchema, predict_schema: GraphSchema, tar_fs: TarFS
    ) -> None:
        for filename, schema in zip(
            [MODEL_ARCHIVE_TRAIN_SCHEMA_FILE, MODEL_ARCHIVE_PREDICT_SCHEMA_FILE],
            [train_schema, predict_schema],
        ):
            serialized = rasa.shared.utils.io.dump_obj_as_yaml_to_string(schema)
            tar_fs.writetext(
                filename, serialized, encoding=rasa.shared.utils.io.DEFAULT_ENCODING
            )

    @staticmethod
    def _add_metadata_to_archive(
        domain: Domain, model_metadata: Dict[Text, Any], tar_fs: TarFS
    ) -> None:
        model_metadata = model_metadata.copy()
        model_metadata[METADATA_DOMAIN_KEY] = domain.as_dict()
        model_metadata[METADATA_TRAINED_AT_KEY] = time.time()
        model_metadata[METADATA_MODEL_ID_KEY] = uuid.uuid4().hex
        model_metadata[METADATA_RASA_VERSION_KEY] = rasa.__version__

        serialized_metadata = rasa.shared.utils.io.json_to_string(model_metadata)

        tar_fs.writetext(
            MODEL_ARCHIVE_METADATA_FILE,
            serialized_metadata,
            encoding=rasa.shared.utils.io.DEFAULT_ENCODING,
        )

    def unpack(
        self, model_archive_path: Text
    ) -> Tuple[GraphSchema, GraphSchema, Domain, Dict[Text, Any]]:
        with fs.tempfs.TempFS() as temp_fs:
            model_file = temp_fs.getsyspath("model.tar.gz")

            fs.move.copy_file(
                fs.path.dirname(model_archive_path),
                fs.path.basename(model_archive_path),
                temp_fs,
                "model.tar.gz",
            )

            # move model content to local disk so we can tar it
            with fs.tarfs.TarFS(model_file) as tar_fs:
                train_schema, predict_schema, = self._read_schemas(tar_fs)
                metadata = self._load_metadata(tar_fs)

                domain = self._extract_domain_from_metadata(metadata)

                fs.copy.copy_dir(
                    tar_fs, MODEL_ARCHIVE_COMPONENTS_DIR, self._storage_path, "/"
                )

                return (
                    train_schema,
                    predict_schema,
                    domain,
                    metadata,
                )

    def _extract_domain_from_metadata(self, metadata: Dict[Text, Any]) -> Domain:
        serialized_domain = metadata.pop(METADATA_DOMAIN_KEY)
        return Domain.from_dict(serialized_domain)

    def _load_metadata(self, filesystem: FS) -> Dict[Text, Any]:
        stringified_metadata = filesystem.readtext(
            MODEL_ARCHIVE_METADATA_FILE, encoding=rasa.shared.utils.io.DEFAULT_ENCODING,
        )
        return json.loads(stringified_metadata)

    def _read_schemas(self, filesystem: FS) -> Tuple[GraphSchema, GraphSchema]:
        train_schema = filesystem.readtext(
            MODEL_ARCHIVE_TRAIN_SCHEMA_FILE,
            encoding=rasa.shared.utils.io.DEFAULT_ENCODING,
        )
        predict_schema = filesystem.readtext(
            MODEL_ARCHIVE_PREDICT_SCHEMA_FILE,
            encoding=rasa.shared.utils.io.DEFAULT_ENCODING,
        )

        return (
            rasa.shared.utils.io.read_yaml(train_schema),
            rasa.shared.utils.io.read_yaml(predict_schema),
        )
