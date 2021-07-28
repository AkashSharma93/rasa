import time
from pathlib import Path

import pytest
from _pytest.tmpdir import TempPathFactory

import rasa.shared.utils.io
from rasa.engine.model_storage import ModelStorage, Resource
from rasa.shared.core.domain import Domain


def test_write_to_and_read(tmp_path: Path):
    test_filename = "file.txt"
    test_file_content = "hi"

    test_sub_filename = "sub_file"
    test_sub_dir_name = "sub_directory"
    test_sub_file_content = "sub file"

    resource = Resource("some_node123")

    model_storage = ModelStorage(tmp_path)

    # Fill model storage for resource
    with model_storage.write_to(resource) as resource_directory:
        file = resource_directory / test_filename
        file.write_text(test_file_content)

        sub_directory = resource_directory / test_sub_dir_name
        sub_directory.mkdir()
        file_in_sub_directory = sub_directory / test_sub_filename
        file_in_sub_directory.write_text(test_sub_file_content)

    # Read written resource data from model storage to see whether all expected
    # content is there
    with model_storage.read_from(resource) as resource_directory:
        assert (resource_directory / test_filename).read_text() == test_file_content
        assert (
            resource_directory / test_sub_dir_name / test_sub_filename
        ).read_text() == test_sub_file_content


def test_read_from_not_existing_resource(tmp_path: Path):
    model_storage = ModelStorage(tmp_path)

    with model_storage.write_to(Resource("resource1")) as temporary_directory:
        file = temporary_directory / "file.txt"
        file.write_text("test")

    with pytest.raises(ValueError):
        with model_storage.read_from(Resource("a different resource")) as _:
            pass


def test_create_model_package(
    tmp_path_factory: TempPathFactory, domain: Domain,
):
    train_model_storage = ModelStorage(tmp_path_factory.mktemp("train model storage"))

    # Fill model Storage
    with train_model_storage.write_to(Resource("resource1")) as directory:
        file = directory / "file.txt"
        file.write_text("test")

    # Package model
    persisted_model_dir = tmp_path_factory.mktemp("persisted models")
    archive_path = persisted_model_dir / "my-model.tar.gz"

    train_schema = {"train_node": [1, 2, 3]}
    predict_schema = {"predict_node": [1, 2, 3]}
    model_metadata = {"some_key": "value"}

    train_model_storage.create_model_package(
        archive_path, train_schema, predict_schema, domain, model_metadata
    )

    # Unpack and inspect packaged model
    load_model_storage_dir = tmp_path_factory.mktemp("load model storage")
    load_model_storage = ModelStorage(load_model_storage_dir)

    (
        packaged_train_schema,
        packaged_predict_schema,
        packaged_domain,
        packaged_metadata,
    ) = load_model_storage.unpack(archive_path)

    assert packaged_train_schema == train_schema
    assert packaged_predict_schema == predict_schema
    assert packaged_domain.as_dict() == domain.as_dict()

    assert packaged_metadata.pop("rasa_open_source_version") == rasa.__version__
    assert float(packaged_metadata.pop("trained_at")) > time.time() - 10
    assert packaged_metadata.pop("model_id")

    assert packaged_metadata == model_metadata

    persisted_resources = load_model_storage_dir.glob("*")
    assert list(persisted_resources) == [Path(load_model_storage_dir, "resource1")]


def test_resource_caching(tmp_path_factory: TempPathFactory):
    model_storage = ModelStorage(str(tmp_path_factory.mktemp("initial_model_storage")))

    resource = Resource("my resource")

    # Fill model storage
    test_filename = "file.txt"
    test_content = "test_resource_caching"
    with model_storage.write_to(resource) as temporary_directory:
        file = temporary_directory / test_filename
        file.write_text(test_content)

    cache_dir = tmp_path_factory.mktemp("cache_dir")

    # Cache resource
    resource.to_cache(cache_dir, model_storage)

    # Reload resource from cache and inspect
    new_model_storage = ModelStorage(str(tmp_path_factory.mktemp("new_model_storage")))
    reinstantiated_resource = Resource.from_cache(
        cache_dir, resource.name, new_model_storage
    )

    assert reinstantiated_resource == resource

    # Read written resource data from model storage to see whether all expected
    # contents are there
    with new_model_storage.read_from(resource) as temporary_directory:
        assert (temporary_directory / test_filename).read_text() == test_content


def test_resource_fingerprinting():
    resource1 = Resource("resource 1")
    resource2 = Resource("resource 2")

    fingerprint1 = resource1.fingerprint()
    fingerprint2 = resource2.fingerprint()

    assert fingerprint1
    assert fingerprint2

    assert fingerprint1 != fingerprint2
