from streamlit.testing.v1 import AppTest

from config import get_settings
from services import qa_service


EXPECTED_BUTTONS = {
    "換一批隨機題",
    "隨機題 1",
    "隨機題 2",
    "隨機題 3",
    "隨機抽一題並解析",
    "帶入問題",
    "帶入並解析",
    "開始解析",
    "清除",
}


def test_streamlit_app_starts_without_runtime_exception() -> None:
    app = AppTest.from_file("app.py").run(timeout=15)

    assert not app.exception
    assert app.title[0].value == "內部法規解析助手"
    assert {button.label for button in app.button} >= EXPECTED_BUTTONS
    assert any(
        item.value == "已載入 100 題。" for item in app.success
    )
    assert app.radio[0].value == "內建 100 題"


def test_clicking_random_question_fills_input_without_calling_model(
    monkeypatch,
) -> None:
    create_service_calls: list[object] = []

    def forbidden_create_service(*args, **kwargs):
        create_service_calls.append((args, kwargs))
        raise AssertionError("選取隨機題時不應建立 QA／模型服務")

    monkeypatch.setattr(
        qa_service,
        "create_qa_service",
        forbidden_create_service,
    )
    app = AppTest.from_file("app.py").run(timeout=15)
    random_button = next(
        button for button in app.button if button.label == "隨機題 1"
    )
    expected_question = random_button.help

    random_button.click().run(timeout=15)

    assert not app.exception
    assert isinstance(expected_question, str)
    assert expected_question
    assert app.text_area(key="question_input").value == expected_question
    assert app.session_state["question_input"] == expected_question
    assert app.session_state["qa_result"] is None
    assert create_service_calls == []


def test_streamlit_app_handles_invalid_environment_without_traceback(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VECTOR_WEIGHT", "0")
    monkeypatch.setenv("KEYWORD_WEIGHT", "0")
    get_settings.cache_clear()

    app = AppTest.from_file("app.py").run(timeout=15)

    assert not app.exception
    assert app.error
    assert "設定檔內容無效" in app.error[0].value
