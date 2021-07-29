from pathlib import Path

import pytest

from rasa.engine.model_storage import ModelStorage


@pytest.fixture()
def default_model_storage(tmp_path: Path) -> ModelStorage:
    return ModelStorage(tmp_path)
