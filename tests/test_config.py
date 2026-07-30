from config import EMBEDDING_DIMENSION, Settings


def test_embedding_dimension_is_not_an_environment_setting(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1536")

    settings = Settings(_env_file=None)

    assert EMBEDDING_DIMENSION == 768
    assert "embedding_dimension" not in Settings.model_fields
    assert not hasattr(settings, "embedding_dimension")
