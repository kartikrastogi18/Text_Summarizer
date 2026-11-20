import pytest
from unittest.mock import MagicMock
import sys

# Fixture to ensure the 'app' module is freshly imported for each test.
# This is crucial when testing global-scope code like model loading.
@pytest.fixture(autouse=True)
def clean_app_module():
    """
    Ensures that the 'app' module is removed from sys.modules before and after each test.
    This allows a fresh import of 'app.py' within the test, ensuring mocks are active
    during module-level code execution (like global variable assignments).
    """
    if 'app' in sys.modules:
        del sys.modules['app']
    yield
    if 'app' in sys.modules:
        del sys.modules['app']

def test_app_initialization_does_not_raise_api_key_error(mocker):
    """
    Tests that the application initializes successfully WITHOUT raising an
    "api key not found" error during the loading of the T5 model.

    This test is designed to FAIL if the reported "api key not found" error occurs
    during application startup (demonstrating the bug), and PASS if it does not
    (indicating the bug has been resolved).
    """
    # Define the specific error message reported in the bug.
    buggy_error_message = "api key not found"

    # --- Mocks for Global Imports and Initializations ---
    # These mocks prevent actual network calls, file system access, or heavy computations
    # during the test, allowing us to control the environment and isolate the bug.

    # Mock FastAPI components to allow `app.py` to import without issues.
    mocker.patch('fastapi.FastAPI', return_value=MagicMock())
    mocker.patch('fastapi.Request', return_value=MagicMock())
    mocker.patch('pydantic.BaseModel', return_value=MagicMock())
    mocker.patch('fastapi.templating.Jinja2Templates', return_value=MagicMock())
    mocker.patch('fastapi.responses.HTMLResponse', return_value=MagicMock())
    mocker.patch('fastapi.staticfiles.StaticFiles', return_value=MagicMock())

    # Mock torch components, as they are part of device detection logic at startup.
    mocker.patch('torch.backends.mps.is_available', return_value=False)
    mocker.patch('torch.cuda.is_available', return_value=False)
    # Mock torch.device to return a simple mock object, as its actual functionality
    # isn't relevant for this specific bug reproduction.
    mocker.patch('torch.device', return_value=MagicMock(name="torch_device_mock"))

    # --- Core Mocks for `transformers.from_pretrained` ---
    # We mock `from_pretrained` calls to *succeed* by default. If the original `app.py`
    # code causes an "api key not found" error despite these mocks, it means the
    # error originates from the actual application code's attempt to load the model.
    mock_model_from_pretrained = mocker.patch(
        'transformers.T5ForConditionalGeneration.from_pretrained',
        return_value=MagicMock(name="mock_t5_model")
    )
    mock_tokenizer_from_pretrained = mocker.patch(
        'transformers.T5Tokenizer.from_pretrained',
        return_value=MagicMock(name="mock_t5_tokenizer")
    )

    # --- Test Execution ---
    # We attempt to import the 'app' module. This action will trigger the execution
    # of the global-scope code in `app.py`, including the model and tokenizer loading.
    try:
        # Assuming 'app.py' is in the same directory or accessible via PYTHONPATH.
        # This import will attempt to load the T5 model and tokenizer.
        import app
        
        # If we reach here, `app.py`'s initialization (including model loading)
        # completed without raising the specific 'api key not found' error.
        # This signifies a successful outcome (bug resolved) for the test.

        # Verify that `from_pretrained` was called with the expected local path.
        mock_model_from_pretrained.assert_called_once_with("./saved_summary_model")
        mock_tokenizer_from_pretrained.assert_called_once_with("./saved_summary_model")

    except OSError as e:
        # If an OSError is caught, we check if it's the specific "api key not found" error.
        # If it is, the bug is still present, and we explicitly fail the test.
        if buggy_error_message in str(e):
            pytest.fail(
                f"App initialization failed with reported bug: '{buggy_error_message}'. "
                f"The test *failed* as expected on buggy code."
            )
        else:
            # If it's a different OSError, let it propagate or fail with a generic message.
            pytest.fail(f"App initialization failed with an unexpected OSError: {e}")
    except Exception as e:
        # Catch any other unexpected exceptions during app import
        pytest.fail(f"App initialization failed with an unexpected general exception: {e}")