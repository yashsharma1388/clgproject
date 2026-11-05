"""
Automated unit and widget tests for Streamlit UI using Streamlit AppTest.
"""

from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_streamlit_app_loads_cleanly():
    """Verify Streamlit app initializes without unhandled runtime exceptions."""
    app_path = Path(__file__).parent.parent / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=15)

    # Verify no error alerts were raised
    assert len(at.error) == 0

    # Verify key UI components rendered
    assert len(at.sidebar.title) >= 1
    assert "System Control" in at.sidebar.title[0].value

    # Verify main title
    assert len(at.title) >= 1
    assert "Multimodal AI Assistant" in at.title[0].value

    # Verify chat input widget exists
    assert len(at.chat_input) >= 1
