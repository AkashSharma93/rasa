import dataclasses
import logging
import shutil
import uuid
from pathlib import Path
from typing import Dict, Text

import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch

import rasa.shared.utils.io
from rasa.engine.caching import (
    TrainingCache,
    CACHE_LOCATION_ENV,
    DEFAULT_CACHE_NAME,
    CACHE_SIZE_ENV,
)
from rasa.engine.model_storage import ModelStorage


@pytest.fixture()
def temp_cache(tmp_path: Path, monkeypatch: MonkeyPatch) -> TrainingCache:
    monkeypatch.setenv(CACHE_LOCATION_ENV, str(tmp_path))
    return TrainingCache()


def _create_test_file_with_size(directory: Path, size_in_mb: float) -> None:
    with open(directory / "fake_file", mode="wb") as f:
        f.seek(int(1024 * 1024 * size_in_mb))
        f.write(b"\0")


@dataclasses.dataclass
class TestCacheableOutput:

    value: Dict
    size_in_mb: int = 0

    def to_cache(self, directory: Path, model_storage: ModelStorage) -> None:
        rasa.shared.utils.io.dump_obj_as_json_to_file(
            directory / "cached.json", self.value
        )

        # Can be used to create cache results of desired size
        if self.size_in_mb:
            _create_test_file_with_size(directory, self.size_in_mb)

    @classmethod
    def from_cache(
        cls, node_name: Text, directory: Path, model_storage: ModelStorage
    ) -> "TestCacheableOutput":
        value = rasa.shared.utils.io.read_json_file(directory / "cached.json")

        return cls(value)


def test_cache_output(tmp_path: Path, temp_cache: TrainingCache):
    model_storage = ModelStorage(tmp_path)

    fingerprint_key = uuid.uuid4().hex
    output = TestCacheableOutput({"something to cache": "dasdaasda"})
    output_fingerprint = uuid.uuid4().hex

    temp_cache.cache_output(fingerprint_key, output, output_fingerprint, model_storage)

    assert (
        temp_cache.get_cached_output_fingerprint(fingerprint_key) == output_fingerprint
    )

    cached_output, output_type = temp_cache.get_cached_result(output_fingerprint)

    assert output_type == TestCacheableOutput
    assert output_type.from_cache("xy", cached_output, model_storage) == output


def test_get_cached_result_with_miss(tmp_path: Path, temp_cache: TrainingCache):
    model_storage = ModelStorage(tmp_path)

    # Cache something
    temp_cache.cache_output(uuid.uuid4().hex, None, uuid.uuid4().hex, model_storage)

    assert temp_cache.get_cached_result(uuid.uuid4().hex) == (None, None)
    assert temp_cache.get_cached_output_fingerprint(uuid.uuid4().hex) is None


def test_get_cached_result_when_result_no_longer_available(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    model_storage = ModelStorage(tmp_path)

    monkeypatch.setenv(CACHE_LOCATION_ENV, str(tmp_path))

    cache = TrainingCache()

    output = TestCacheableOutput({"something to cache": "dasdaasda"})
    output_fingerprint = uuid.uuid4().hex

    cache.cache_output(uuid.uuid4().hex, output, output_fingerprint, model_storage)

    # Pretend something deleted the cache in between
    for path in tmp_path.glob("*"):
        if path.is_dir():
            shutil.rmtree(path)

    assert cache.get_cached_result(output_fingerprint) == (None, None)


def test_cache_creates_location_if_missing(tmp_path: Path, monkeypatch: MonkeyPatch):
    cache_location = tmp_path / "directory does not exist yet"

    monkeypatch.setenv(CACHE_LOCATION_ENV, str(cache_location))

    _ = TrainingCache()

    assert cache_location.is_dir()


def test_caching_something_which_is_not_cacheable(
    tmp_path: Path, temp_cache: TrainingCache
):
    model_storage = ModelStorage(tmp_path)

    # Cache something
    fingerprint_key = uuid.uuid4().hex
    output_fingerprint_key = uuid.uuid4().hex
    temp_cache.cache_output(
        fingerprint_key, None, output_fingerprint_key, model_storage
    )

    # Output fingerprint was saved
    assert (
        temp_cache.get_cached_output_fingerprint(fingerprint_key)
        == output_fingerprint_key
    )

    # But it's not stored to disk
    assert temp_cache.get_cached_result(output_fingerprint_key) == (None, None)


def test_caching_cacheable_fails(
    tmp_path: Path, caplog: LogCaptureFixture, temp_cache: TrainingCache
):
    model_storage = ModelStorage(tmp_path)

    fingerprint_key = uuid.uuid4().hex

    # `tmp_path` is not a dict and will hence fail to be cached
    # noinspection PyTypeChecker
    output = TestCacheableOutput(tmp_path)
    output_fingerprint = uuid.uuid4().hex

    with caplog.at_level(logging.ERROR):
        temp_cache.cache_output(
            fingerprint_key, output, output_fingerprint, model_storage
        )

    assert len(caplog.records) == 1

    assert (
        temp_cache.get_cached_output_fingerprint(fingerprint_key) == output_fingerprint
    )

    assert temp_cache.get_cached_result(output_fingerprint) == (None, None)


def test_removing_no_longer_compatible_cache_entries(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    model_storage = ModelStorage(tmp_path)
    monkeypatch.setenv(CACHE_LOCATION_ENV, str(tmp_path))

    cache = TrainingCache()

    # Cache metadata and data
    fingerprint_key1 = uuid.uuid4().hex
    output1 = TestCacheableOutput({"something to cache": "dasdaasda"})
    output_fingerprint1 = uuid.uuid4().hex

    cache.cache_output(fingerprint_key1, output1, output_fingerprint1, model_storage)

    # Cache only metadata (`output` is not `Cacheable`)
    fingerprint_key2 = uuid.uuid4().hex
    output_fingerprint2 = uuid.uuid4().hex
    cache.cache_output(fingerprint_key2, None, output_fingerprint2, model_storage)

    # Pretend we updated Rasa Open Source to a no longer compatible version
    monkeypatch.setattr(rasa.engine.caching, "MINIMUM_COMPATIBLE_VERSION", "99999.9.9")

    cache_run_by_future_rasa = TrainingCache()

    # Cached output of no longer compatible stuff was deleted
    assert list(tmp_path.glob("*")) == [tmp_path / DEFAULT_CACHE_NAME]

    # Cached fingerprints can no longer be retrieved
    assert (
        cache_run_by_future_rasa.get_cached_output_fingerprint(fingerprint_key1) is None
    )
    assert (
        cache_run_by_future_rasa.get_cached_output_fingerprint(fingerprint_key2) is None
    )

    assert cache_run_by_future_rasa.get_cached_result(output_fingerprint1) == (
        None,
        None,
    )
    assert cache_run_by_future_rasa.get_cached_result(output_fingerprint2) == (
        None,
        None,
    )


def test_skip_caching_if_cache_size_is_zero(tmp_path: Path, monkeypatch: MonkeyPatch):
    model_storage = ModelStorage(tmp_path)
    monkeypatch.setenv(CACHE_LOCATION_ENV, str(tmp_path))

    # Disable cache
    monkeypatch.setenv(CACHE_SIZE_ENV, "0")

    cache = TrainingCache()

    # Cache something
    fingerprint_key1 = uuid.uuid4().hex
    output1 = TestCacheableOutput({"something to cache": "dasdaasda"})
    output_fingerprint1 = uuid.uuid4().hex
    cache.cache_output(fingerprint_key1, output1, output_fingerprint1, model_storage)

    # not even the database was created ⛔️
    assert list(tmp_path.glob("*")) == []

    assert cache.get_cached_output_fingerprint(fingerprint_key1) is None

    assert cache.get_cached_result(output_fingerprint1) == (None, None,)


def test_skip_caching_if_result_exceeds_max_size(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    model_storage = ModelStorage(tmp_path)
    monkeypatch.setenv(CACHE_LOCATION_ENV, str(tmp_path))

    # Pretend we have a cache of size "1"
    monkeypatch.setenv(CACHE_SIZE_ENV, "1")

    cache = TrainingCache()

    # Cache something
    fingerprint_key1 = uuid.uuid4().hex
    output1 = TestCacheableOutput({"something to cache": "dasdaasda"}, size_in_mb=2)
    output_fingerprint1 = uuid.uuid4().hex
    cache.cache_output(fingerprint_key1, output1, output_fingerprint1, model_storage)

    assert cache.get_cached_output_fingerprint(fingerprint_key1) == output_fingerprint1

    assert cache.get_cached_result(output_fingerprint1) == (None, None)


def test_delete_using_lru_if_cache_exceeds_size(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    model_storage = ModelStorage(tmp_path)
    monkeypatch.setenv(CACHE_LOCATION_ENV, str(tmp_path))

    # Pretend we have a cache of size "1"
    monkeypatch.setenv(CACHE_SIZE_ENV, "5")

    cache = TrainingCache()

    # Cache an item
    fingerprint_key1 = uuid.uuid4().hex
    output1 = TestCacheableOutput({"something to cache": "dasdaasda"}, size_in_mb=2)
    output_fingerprint1 = uuid.uuid4().hex
    cache.cache_output(fingerprint_key1, output1, output_fingerprint1, model_storage)

    # Cache an non cacheable item to spice it up 🔥
    fingerprint_key2 = uuid.uuid4().hex
    output2 = TestCacheableOutput(None)
    output_fingerprint2 = uuid.uuid4().hex
    cache.cache_output(fingerprint_key2, output2, output_fingerprint2, model_storage)

    # Cache another item
    fingerprint_key3 = uuid.uuid4().hex
    output3 = TestCacheableOutput({"something to cache": "dasdaasda"}, size_in_mb=2)
    output_fingerprint3 = uuid.uuid4().hex
    cache.cache_output(fingerprint_key3, output3, output_fingerprint3, model_storage)

    # Assert both are there
    for output_fingerprint in [output_fingerprint1, output_fingerprint2]:
        dir, _ = cache.get_cached_result(output_fingerprint)
        assert dir

    # Checkout the first item as this updates `last_used` and hence affects LRU
    cache.get_cached_output_fingerprint(fingerprint_key1)

    # Now store something which requires a deletion
    fingerprint_key4 = uuid.uuid4().hex
    output4 = TestCacheableOutput({"something to cache": "dasdaasda"}, size_in_mb=2)
    output_fingerprint4 = uuid.uuid4().hex
    cache.cache_output(fingerprint_key4, output4, output_fingerprint4, model_storage)

    # Assert cached result 1 and 3 are there
    for output_fingerprint in [output_fingerprint1, output_fingerprint4]:
        dir, _ = cache.get_cached_result(output_fingerprint)
        assert dir

    # Cached result 2 and 3 were deleted
    assert cache.get_cached_output_fingerprint(fingerprint_key2) is None
    assert cache.get_cached_result(output_fingerprint3) == (None, None)
