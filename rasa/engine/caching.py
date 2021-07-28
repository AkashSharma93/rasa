import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Text, Any, Optional, Tuple, Type

from sqlalchemy.engine import URL

# TODO: Make this a prod dependency
from typing_extensions import Protocol, runtime_checkable

import rasa
import rasa.shared.utils.common
from rasa.engine.model_storage import ModelStorage
import sqlalchemy as sa
import sqlalchemy.orm

logger = logging.getLogger(__name__)

DEFAULT_CACHE_LOCATION = Path(".rasa", "cache")
DEFAULT_CACHE_NAME = "cache.db"

CACHE_LOCATION_ENV = "RASA_CACHE_DIRECTORY"
CACHE_DB_NAME_ENV = "RASA_CACHE_NAME"


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

        self._cache_database_name = os.environ.get(
            CACHE_DB_NAME_ENV, DEFAULT_CACHE_NAME
        )

        engine = sa.create_engine(
            URL.create(
                drivername="sqlite",
                database=str(self._cache_location / self._cache_database_name),
            )
        )
        self.Base.metadata.create_all(engine)
        self._sessionmaker = sa.orm.sessionmaker(engine)

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

    def _cache_output_to_disk(
        self, output: Cacheable, model_storage: ModelStorage
    ) -> Tuple[Optional[Text], Optional[Text]]:
        cache_dir = self._cache_location / uuid.uuid4().hex
        cache_dir.mkdir()
        try:
            output.to_cache(cache_dir, model_storage)
            output_type = rasa.shared.utils.common.module_path_from_instance(output)

            logger.debug(f"Caching output of type '{type(output)}' succeeded.")
            return str(cache_dir.absolute()), output_type
        except Exception as e:
            logger.error(
                f"Caching output of type '{type(output)}' failed with the "
                f"following error:\n{e}"
            )
            return None, None

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
            query = sa.select(self.CacheEntry.output_fingerprint_key).filter_by(
                fingerprint_key=fingerprint_key
            )
            match = session.execute(query).first()

            if match:
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
