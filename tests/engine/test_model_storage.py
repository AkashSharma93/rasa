import abc
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Text

import fs.move
import pytest
from _pytest.tmpdir import TempPathFactory

import rasa.shared.utils.io
from rasa.engine.model_storage import ModelStorage, Resource
from rasa.shared.core.domain import Domain


class ModelStorageTest(abc.ABC):
    @pytest.fixture()
    @abc.abstractmethod
    def archive_url(self) -> Text:
        pass

    def test_write_to_and_read(self, archive_url: Text):
        test_filename = "file.txt"
        test_file_content = "hi"

        test_sub_filename = "sub_file"
        test_sub_dir_name = "sub_directory"
        test_sub_file_content = "sub file"

        resource = Resource("some_node123")

        model_storage = ModelStorage.create_for(archive_url)

        # Fill model storage for resource
        with model_storage.write_to(resource) as temporary_directory:
            file = temporary_directory / test_filename
            file.write_text(test_file_content)

            sub_directory = temporary_directory / test_sub_dir_name
            sub_directory.mkdir()
            file_in_sub_directory = sub_directory / test_sub_filename
            file_in_sub_directory.write_text(test_sub_file_content)

        # Read written resource data from model storage to see whether all expected
        # contents are there
        with model_storage.read_from(resource) as temporary_directory:
            assert (
                temporary_directory / test_filename
            ).read_text() == test_file_content
            assert (
                temporary_directory / test_sub_dir_name / test_sub_filename
            ).read_text() == test_sub_file_content

    def test_read_from_not_existing_resource(self, archive_url: Text):
        model_storage = ModelStorage.create_for(archive_url)

        with model_storage.write_to(Resource("resource1")) as temporary_directory:
            file = temporary_directory / "file.txt"
            file.write_text("test")

        with model_storage.read_from(
            Resource("a different resource")
        ) as temporary_directory:
            assert list(temporary_directory.glob("*")) == []

    def test_create_model_package(
        self, archive_url: Text, tmp_path: Path, domain: Domain
    ):
        model_storage = ModelStorage.create_for(archive_url)

        # Fill model Storage
        with model_storage.write_to(Resource("resource1")) as temporary_directory:
            file = temporary_directory / "file.txt"
            file.write_text("test")

        # Package model
        train_schema = {"train_node": [1, 2, 3]}
        predict_schema = {"predict_node": [1, 2, 3]}
        model_metadata = {"some_key": "value"}
        model_storage.create_model_package(
            train_schema, predict_schema, domain, model_metadata
        )

        # Copy model archive to local disk so we can investigate it
        local_model_package = tmp_path / "my_archive.tar.gz"
        fs.move.move_file(
            fs.path.dirname(archive_url),
            fs.path.basename(archive_url),
            str(tmp_path),
            "my_archive.tar.gz",
        )

        unpacked_directory = tmp_path / "unpackaged"
        unpacked_directory.mkdir()
        new_model_file = unpacked_directory / "some-model.tar.gz"
        shutil.move(local_model_package, new_model_file)
        new_model_storage = ModelStorage.create_for(str(new_model_file))

        (
            packaged_train_schema,
            packaged_predict_schema,
            packaged_domain,
            packaged_metadata,
        ) = new_model_storage.unpack()

        assert packaged_train_schema == train_schema
        assert packaged_predict_schema == predict_schema
        assert packaged_domain.as_dict() == domain.as_dict()

        assert packaged_metadata.pop("rasa_open_source_version") == rasa.__version__
        assert float(packaged_metadata.pop("trained_at")) > time.time() - 10
        assert packaged_metadata.pop("model_id")

        assert packaged_metadata == model_metadata

        persisted_resources = (unpacked_directory / "components").glob("*")
        assert list(persisted_resources) == [
            Path(unpacked_directory, "components", "resource1")
        ]


class TestLocalModelStorage(ModelStorageTest):
    @pytest.fixture()
    def archive_url(self) -> Text:
        with tempfile.TemporaryDirectory() as dir:
            yield str(Path(dir) / "my_model.tar.gz")


class TestInmemoryModelStorage(ModelStorageTest):
    @pytest.fixture()
    def archive_url(self) -> Text:
        return "mem:///my_model.tar.gz"


def test_resource_caching(tmp_path_factory: TempPathFactory):
    model_archive = tmp_path_factory.mktemp("initial_model_storage") / "test.tar.gz"
    model_storage = ModelStorage.create_for(str(model_archive))

    resource = Resource("my resource")

    # Fill model storage
    test_filename = "file.txt"
    test_content = "test_resource_caching"
    with model_storage.write_to(resource) as temporary_directory:
        file = temporary_directory / test_filename
        file.write_text(test_content)

    cache_dir = tmp_path_factory.mktemp("cache_dir")

    resource.to_cache(cache_dir, model_storage)

    new_model_archive = (
        tmp_path_factory.mktemp("new_model_storage") / "new_model.tar.gz"
    )
    new_model_storage = ModelStorage.create_for(str(new_model_archive))
    reinstantiated_resource = Resource.from_cache(
        cache_dir, resource.name, new_model_storage
    )

    assert reinstantiated_resource == resource

    # Read written resource data from model storage to see whether all expected
    # contents are there
    with new_model_storage.read_from(resource) as temporary_directory:
        assert (temporary_directory / test_filename).read_text() == test_content
