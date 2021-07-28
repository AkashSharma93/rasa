import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Text, Any, Optional, Tuple, Type, List

from packaging import version
from sqlalchemy.engine import URL

# TODO: Make this a prod dependency
from typing_extensions import Protocol, runtime_checkable

import rasa
import rasa.shared.utils.common
from rasa.constants import MINIMUM_COMPATIBLE_VERSION
from rasa.engine.model_storage import ModelStorage
import sqlalchemy as sa
import sqlalchemy.orm

logger = logging.getLogger(__name__)

DEFAULT_CACHE_LOCATION = Path(".rasa", "cache")
DEFAULT_CACHE_NAME = "cache.db"
DEFAULT_CACHE_SIZE_MB = 1000

CACHE_LOCATION_ENV = "RASA_CACHE_DIRECTORY"
CACHE_DB_NAME_ENV = "RASA_CACHE_NAME"
CACHE_SIZE_ENV = "RASA_MAX_CACHE_SIZE"


@runtime_checkable
class Cacheable(Protocol):
    """Protocol for cacheable graph component outputs.

    We only cache graph component outputs which are `Cacheable`. We only store the
    output fingerprint for everything else.
    """

    def to_cache(self, directory: Path, model_storage: ModelStorage) -> None:
        """Persists `Cacheable` to disk.

        Args:
            directory: The directory where the `Cacheable` can persist itself to.
            model_storage: The current model storage (e.g. used when caching `Resource`
                objects.
        """
        ...

    @classmethod
    def from_cache(
        cls, node_name: Text, directory: Path, model_storage: ModelStorage
    ) -> "Cacheable":
        """Loads `Cacheable` from cache.

        Args:
            node_name: The name of the graph node which wants to use this cached result.
            directory: Directory containing the persisted `Cacheable`.
            model_storage: The current model storage (e.g. used when restoring
                `Resource` objects so that they can fill the model storage with data.

        Returns:
            Instantiated `Cacheable`.
        """
        ...


class TrainingCache:
    """Stores training results in a persistent cache.

    Used to minimize re-retraining when the data / config didn't change in between
    training runs.
    """

    from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta

    Base: DeclarativeMeta = declarative_base()

    class CacheEntry(Base):
        """Stores metadata about a single cache entry."""

        __tablename__ = "cache_entry"

        # `create_sequence` is needed to create a sequence for databases that
        # don't autoincrement Integer primary keys (e.g. Oracle)
        fingerprint_key = sa.Column(sa.String(), primary_key=True)
        output_fingerprint_key = sa.Column(sa.String(), nullable=False, index=True)
        last_used = sa.Column(sa.DateTime(), nullable=False)
        rasa_version = sa.Column(sa.String(255), nullable=False)
        result_location = sa.Column(sa.String())
        result_type = sa.Column(sa.String())

    def __init__(self) -> None:
        """Creates cache.

        The `Cache` setting can be configured via environment variables.
        """
        self._cache_location = Path(
            os.environ.get(CACHE_LOCATION_ENV, DEFAULT_CACHE_LOCATION)
        )

        if not self._cache_location.exists():
            self._cache_location.mkdir(parents=True)

        self._max_cache_size = float(
            os.environ.get(CACHE_SIZE_ENV, DEFAULT_CACHE_SIZE_MB)
        )
        self._cache_database_name = os.environ.get(
            CACHE_DB_NAME_ENV, DEFAULT_CACHE_NAME
        )

        self._sessionmaker = self._create_database()

        self._drop_cache_entries_from_incompatible_versions()

    def _create_database(self) -> sqlalchemy.orm.sessionmaker:
        if self._is_disabled():
            # Use in-memory database as mock to avoid having to check `_is_disabled`
            # everywhere
            database = ""
        else:
            database = str(self._cache_location / self._cache_database_name)

        engine = sa.create_engine(URL.create(drivername="sqlite", database=database,))
        self.Base.metadata.create_all(engine)

        return sa.orm.sessionmaker(engine)

    def _drop_cache_entries_from_incompatible_versions(self) -> None:
        with self._sessionmaker() as session:
            query_for_cache_entries = sa.select(self.CacheEntry)
            all_entries: List[TrainingCache.CacheEntry] = session.execute(
                query_for_cache_entries
            ).scalars().all()

        incompatible_entries = [
            entry
            for entry in all_entries
            if version.parse(MINIMUM_COMPATIBLE_VERSION)
            > version.parse(entry.rasa_version)
        ]

        for entry in incompatible_entries:
            self._delete_persisted(entry)

        incompatible_fingerprints = [
            entry.fingerprint_key for entry in incompatible_entries
        ]
        with self._sessionmaker.begin() as session:
            delete_query = sa.delete(self.CacheEntry).where(
                self.CacheEntry.fingerprint_key.in_(incompatible_fingerprints)
            )
            session.execute(delete_query)

        logger.debug(
            f"Deleted {len(incompatible_entries)} from disk as their version "
            f"is smaller than the minimum compatible version "
            f"('{MINIMUM_COMPATIBLE_VERSION}')."
        )

    @staticmethod
    def _delete_persisted(entry: "TrainingCache.CacheEntry") -> None:
        if entry.result_location and Path(entry.result_location).is_dir():
            shutil.rmtree(entry.result_location)

    def cache_output(
        self,
        fingerprint_key: Text,
        output: Any,
        output_fingerprint: Text,
        model_storage: ModelStorage,
    ) -> None:
        """Adds the output to the cache.

        If the output is of type `Cacheable` the output is persisted to disk in addition
        to its fingerprint.

        Args:
            fingerprint_key: The fingerprint key serves as key for the cache. Graph
                components can use their fingerprint key to lookup fingerprints of
                previous training runs.
            output: The output. The output is only cached to disk if it's of type
                `Cacheable`.
            output_fingerprint: The fingerprint of they output. This can be used
                to lookup potentially persisted outputs on disk.
            model_storage: Required for caching `Cacheable` instances. E.g. `Resource`s
                use that to copy data from the model storage to the cache.
        """
        if self._is_disabled():
            return

        cache_dir, output_type = None, None
        if isinstance(output, Cacheable):
            cache_dir, output_type = self._cache_output_to_disk(output, model_storage)

        with self._sessionmaker.begin() as session:
            cache_entry = self.CacheEntry(
                fingerprint_key=fingerprint_key,
                output_fingerprint_key=output_fingerprint,
                last_used=datetime.utcnow(),
                rasa_version=rasa.__version__,
                result_location=cache_dir,
                result_type=output_type,
            )
            session.add(cache_entry)

    def _is_disabled(self) -> bool:
        return self._max_cache_size == 0.0

    def _cache_output_to_disk(
        self, output: Cacheable, model_storage: ModelStorage
    ) -> Tuple[Optional[Text], Optional[Text]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                output.to_cache(Path(temp_dir), model_storage)

                logger.debug(f"Caching output of type '{type(output)}' succeeded.")
            except Exception as e:
                logger.error(
                    f"Caching output of type '{type(output)}' failed with the "
                    f"following error:\n{e}"
                )
                return None, None

            output_size = _directory_size_in_mb(temp_dir)
            if output_size > self._max_cache_size:
                logger.debug(
                    f"Caching result of type '{type(output)}' was skipped "
                    f"because it exceeds the maximum cache size of "
                    f"{self._max_cache_size} MiB."
                )
                return None, None

            while (
                _directory_size_in_mb(self._cache_location) + output_size
                > self._max_cache_size
            ):
                self._drop_least_recently_used_item()

            output_type = rasa.shared.utils.common.module_path_from_instance(output)
            cache_path = shutil.move(temp_dir, self._cache_location)

            return cache_path, output_type

    def _drop_least_recently_used_item(self) -> None:
        with self._sessionmaker.begin() as session:
            query_for_least_recently_used_entry = sa.select(self.CacheEntry).order_by(
                self.CacheEntry.last_used.asc()
            )
            oldest_cache_item = (
                session.execute(query_for_least_recently_used_entry).scalars().first()
            )

            # TODO: Can't happen
            if not oldest_cache_item:
                return

            self._delete_persisted(oldest_cache_item)
            delete_query = sa.delete(self.CacheEntry).where(
                self.CacheEntry.fingerprint_key == oldest_cache_item.fingerprint_key
            )
            session.execute(delete_query)

            logger.debug(
                f"Deleted item with fingerprint "
                f"'{oldest_cache_item.fingerprint_key}' to free space."
            )

    def get_cached_output_fingerprint(self, fingerprint_key: Text) -> Optional[Text]:
        """Retrieves fingerprint of output based on fingerprint key.

        Args:
            fingerprint_key: The fingerprint serves as key for the lookup of output
                fingerprints.

        Returns:
            The fingerprint of a matching output or `None` in case no cache entry was
            found for the given fingerprint key.
        """
        with self._sessionmaker.begin() as session:
            query = sa.select(self.CacheEntry).filter_by(
                fingerprint_key=fingerprint_key
            )
            match = session.execute(query).scalars().first()

            if match:
                # This result was used during a fingerprint run.
                match.last_used = datetime.utcnow()
                return match.output_fingerprint_key

            return None

    def get_cached_result(
        self, output_fingerprint_key: Text
    ) -> Tuple[Optional[Path], Optional[Type[Cacheable]]]:
        """Returns a potentially cached output result.

        Args:
            output_fingerprint_key:

        Returns:
            - Path to directory containing the cached content or `None` in case no
            persisted data was found
            - Class of persisted data or None in case no persisted data was found.
        """
        with self._sessionmaker.begin() as session:
            query = sa.select(
                self.CacheEntry.result_location, self.CacheEntry.result_type
            ).where(
                self.CacheEntry.output_fingerprint_key == output_fingerprint_key,
                self.CacheEntry.result_location != sa.null(),
            )
            match = session.execute(query).first()

        if not match:
            logger.debug(f"No cached output found for '{output_fingerprint_key}'")
            return None, None

        path_to_cached = Path(match.result_location)
        if not path_to_cached.is_dir():
            logger.debug(
                f"Cached output for '{output_fingerprint_key}' can't be found on disk."
            )
            return None, None

        module_as_string = match.result_type
        try:
            module = rasa.shared.utils.common.class_from_module_path(module_as_string)
            assert isinstance(module, Cacheable)

            return path_to_cached, module
        except Exception as e:
            logger.warning(
                f"Failed to find module for cached output of type "
                f"'{module_as_string}'. Error:\n{e}"
            )
            return None, None


def _directory_size_in_mb(path: Path) -> float:
    size = 0.0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            size += (Path(root) / filename).stat().st_size

    # bytes to MiB
    return size / 1_048_576
