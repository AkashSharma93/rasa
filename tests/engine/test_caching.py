from rasa.engine.caching import TrainingCache


def test_cache_output():
    cache = TrainingCache()

    fingerprint_key = "1234"
    output = 1234

    cache.cache_output(fingerprint_key, output, output_fingerprint, model_storage)
