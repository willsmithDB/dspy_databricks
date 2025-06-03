import os
from unittest.mock import Mock, patch

import pytest
import importlib

from dspy.retrieve.databricks_rm import DatabricksRM, Document, _get_oauth_token


class TestDocument:
    """Test cases for the Document dataclass."""

    def test_document_creation(self):
        """Test Document creation and to_dict method."""
        doc = Document(
            page_content="Test content",
            metadata={"key": "value"},
            type="Document"
        )

        assert doc.page_content == "Test content"
        assert doc.metadata == {"key": "value"}
        assert doc.type == "Document"

        expected_dict = {
            "page_content": "Test content",
            "metadata": {"key": "value"},
            "type": "Document"
        }
        assert doc.to_dict() == expected_dict


class TestDatabricksRM:
    """Test cases for the DatabricksRM class."""

    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables."""
        with patch.dict(os.environ, {
            'DATABRICKS_TOKEN': 'test_token',
            'DATABRICKS_HOST': 'https://test.databricks.com',
            'DATABRICKS_CLIENT_ID': 'test_client_id',
            'DATABRICKS_CLIENT_SECRET': 'test_client_secret'
        }):
            yield

    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing."""
        return {
            "manifest": {
                "columns": [
                    {"name": "id"},
                    {"name": "text"},
                    {"name": "score"},
                    {"name": "metadata"}
                ]
            },
            "result": {
                "data_array": [
                    [1, "First document", 0.95, '{"source": "doc1"}'],
                    [2, "Second document", 0.85, '{"source": "doc2"}'],
                    [3, "Third document", 0.75, '{"source": "doc3"}']
                ]
            }
        }

    def test_init_with_explicit_params(self):
        """Test initialization with explicit parameters."""
        retriever = DatabricksRM(
            databricks_index_name="test_index",
            databricks_endpoint="https://test.databricks.com",
            databricks_token="test_token",
            k=5,
            docs_id_column_name="doc_id",
            text_column_name="content"
        )

        assert retriever.databricks_index_name == "test_index"
        assert retriever.databricks_endpoint == "https://test.databricks.com"
        assert retriever.databricks_token == "test_token"
        assert retriever.k == 5
        assert retriever.docs_id_column_name == "doc_id"
        assert retriever.text_column_name == "content"

    def test_init_with_env_vars(self, mock_env_vars):
        """Test initialization using environment variables."""
        retriever = DatabricksRM(databricks_index_name="test_index")

        assert retriever.databricks_token == "test_token"
        assert retriever.databricks_endpoint == "https://test.databricks.com"
        assert retriever.databricks_client_id == "test_client_id"
        assert retriever.databricks_client_secret == "test_client_secret"

    @patch('importlib.util.find_spec')
    @patch('mlflow.models.set_retriever_schema')
    def test_init_with_agent_framework(self, mock_set_schema, mock_find_spec):
        """Test initialization with Databricks Agent Framework."""
        mock_find_spec.return_value = Mock()

        retriever = DatabricksRM(
            databricks_index_name="test_index",
            databricks_token="test_token",
            databricks_endpoint="https://test.databricks.com",
            use_with_databricks_agent_framework=True
        )

        mock_set_schema.assert_called_once_with(
            primary_key="doc_id",
            text_column="page_content",
            doc_uri="doc_uri"
        )

    @patch('importlib.util.find_spec')
    def test_init_agent_framework_missing_mlflow(self, mock_find_spec):
        """Test initialization failure when mlflow not available for agent framework."""
        mock_find_spec.return_value = Mock()

        with patch('builtins.__import__', side_effect=ImportError):
            with pytest.raises(ValueError, match="you must install the mlflow Python library"):
                DatabricksRM(
                    databricks_index_name="test_index",
                    databricks_token="test_token",
                    databricks_endpoint="https://test.databricks.com",
                    use_with_databricks_agent_framework=True
                )

    def test_extract_doc_ids_regular_column(self):
        """Test document ID extraction from regular column."""
        retriever = DatabricksRM(
            databricks_index_name="test_index",
            databricks_token="test_token",
            databricks_endpoint="https://test.databricks.com",
            docs_id_column_name="id"
        )

        item = {"id": "doc123", "text": "content"}
        assert retriever._extract_doc_ids(item) == "doc123"

    def test_extract_doc_ids_metadata_column(self):
        """Test document ID extraction from metadata column."""
        retriever = DatabricksRM(
            databricks_index_name="test_index",
            databricks_token="test_token",
            databricks_endpoint="https://test.databricks.com",
            docs_id_column_name="metadata"
        )

        item = {"metadata": '{"document_id": "doc456", "other": "value"}'}
        assert retriever._extract_doc_ids(item) == "doc456"

    def test_get_extra_columns(self):
        """Test extraction of extra columns."""
        retriever = DatabricksRM(
            databricks_index_name="test_index",
            databricks_token="test_token",
            databricks_endpoint="https://test.databricks.com",
            docs_id_column_name="id",
            text_column_name="text",
            docs_uri_column_name="uri"
        )

        item = {
            "id": "doc123",
            "text": "content",
            "uri": "http://example.com",
            "score": 0.95,
            "category": "test"
        }

        extra = retriever._get_extra_columns(item)
        expected = {"score": 0.95, "category": "test"}
        assert extra == expected

    def test_get_extra_columns_with_metadata(self):
        """Test extraction of extra columns with metadata parsing."""
        retriever = DatabricksRM(
            databricks_index_name="test_index",
            databricks_token="test_token",
            databricks_endpoint="https://test.databricks.com",
            docs_id_column_name="metadata",
            text_column_name="text"
        )

        item = {
            "metadata": '{"document_id": "doc123", "source": "web", "category": "news"}',
            "text": "content",
            "score": 0.95
        }

        extra = retriever._get_extra_columns(item)
        expected = {
            "score": 0.95,
            "metadata": {"source": "web", "category": "news"}
        }
        assert extra == expected

    @patch('dspy.retrieve.databricks_rm._databricks_sdk_installed', True)
    @patch('dspy.retrieve.databricks_rm.DatabricksRM._query_via_databricks_sdk')
    def test_forward_text_query_with_sdk(self, mock_query_sdk, sample_search_results):
        """Test forward method with text query using SDK."""
        mock_query_sdk.return_value = sample_search_results

        retriever = DatabricksRM(
            databricks_index_name="test_index",
            databricks_token="test_token",
            databricks_endpoint="https://test.databricks.com"
        )

        result = retriever.forward("test query")

        mock_query_sdk.assert_called_once()
        assert hasattr(result, 'docs')
        assert len(result.docs) == 3
        assert result.docs[0] == "First document"

    @patch('dspy.retrieve.databricks_rm._databricks_sdk_installed', False)
    @patch('dspy.retrieve.databricks_rm.DatabricksRM._query_via_requests')
    def test_forward_vector_query_with_requests(self, mock_query_requests, sample_search_results):
        """Test forward method with vector query using requests."""
        mock_query_requests.return_value = sample_search_results

        retriever = DatabricksRM(
            databricks_index_name="test_index",
            databricks_token="test_token",
            databricks_endpoint="https://test.databricks.com"
        )

        result = retriever.forward([0.1, 0.2, 0.3])

        mock_query_requests.assert_called_once()
        assert hasattr(result, 'docs')
        assert len(result.docs) == 3

    def test_forward_invalid_query_type(self):
        """Test forward method with invalid query type."""
        retriever = DatabricksRM(
            databricks_index_name="test_index",
            databricks_token="test_token",
            databricks_endpoint="https://test.databricks.com"
        )

        with pytest.raises(ValueError, match="Query must be a string or a list of floats"):
            retriever.forward(123)

    def test_forward_legacy_query_types(self, sample_search_results):
        """Test forward method with legacy query types."""
        with patch('dspy.retrieve.databricks_rm._databricks_sdk_installed', False), \
                patch('dspy.retrieve.databricks_rm.DatabricksRM._query_via_requests') as mock_query:
            mock_query.return_value = sample_search_results

            retriever = DatabricksRM(
                databricks_index_name="test_index",
                databricks_token="test_token",
                databricks_endpoint="https://test.databricks.com"
            )

            # Test legacy "text" query type
            retriever.forward("test query", query_type="text")

            # Verify that "ANN" was used instead of "text"
            args, kwargs = mock_query.call_args
            assert kwargs['query_type'] == "ANN"

    @patch('dspy.retrieve.databricks_rm._databricks_sdk_installed', False)
    @patch('dspy.retrieve.databricks_rm.DatabricksRM._query_via_requests')
    def test_forward_with_agent_framework(self, mock_query_requests, sample_search_results):
        """Test forward method with agent framework enabled."""
        mock_query_requests.return_value = sample_search_results

        retriever = DatabricksRM(
            databricks_index_name="test_index",
            databricks_token="test_token",
            databricks_endpoint="https://test.databricks.com",
            use_with_databricks_agent_framework=True
        )

        result = retriever.forward("test query")

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(doc, dict) for doc in result)
        assert result[0]["page_content"] == "First document"
        assert result[0]["type"] == "Document"

    def test_forward_missing_columns_error(self):
        """Test forward method with missing required columns."""
        invalid_results = {
            "manifest": {
                "columns": [{"name": "wrong_column"}]
            },
            "result": {"data_array": []}
        }

        with patch('dspy.retrieve.databricks_rm._databricks_sdk_installed', False), \
                patch('dspy.retrieve.databricks_rm.DatabricksRM._query_via_requests') as mock_query:
            mock_query.return_value = invalid_results

            retriever = DatabricksRM(
                databricks_index_name="test_index",
                databricks_token="test_token",
                databricks_endpoint="https://test.databricks.com"
            )

            with pytest.raises(Exception, match="docs_id_column_name: 'id' is not in the index columns"):
                retriever.forward("test query")

    @patch('databricks.sdk.WorkspaceClient')
    def test_query_via_databricks_sdk(self, mock_workspace_client):
        """Test querying via Databricks SDK."""
        mock_client = Mock()
        mock_workspace_client.return_value = mock_client
        mock_client.vector_search_indexes.query_index.return_value.as_dict.return_value = {"result": "success"}

        result = DatabricksRM._query_via_databricks_sdk(
            index_name="test_index",
            k=3,
            columns=["id", "text"],
            query_type="ANN",
            query_text="test query",
            query_vector=None,
            databricks_token="test_token",
            databricks_endpoint="https://test.databricks.com",
            databricks_client_id=None,
            databricks_client_secret=None,
            filters_json=None
        )

        assert result == {"result": "success"}
        mock_client.vector_search_indexes.query_index.assert_called_once()

    @patch('databricks.sdk.WorkspaceClient')
    def test_query_via_databricks_sdk_with_service_principal(self, mock_workspace_client):
        """Test querying via Databricks SDK with service principal."""
        mock_client = Mock()
        mock_workspace_client.return_value = mock_client
        mock_client.vector_search_indexes.query_index.return_value.as_dict.return_value = {"result": "success"}

        DatabricksRM._query_via_databricks_sdk(
            index_name="test_index",
            k=3,
            columns=["id", "text"],
            query_type="ANN",
            query_text="test query",
            query_vector=None,
            databricks_token=None,
            databricks_endpoint=None,
            databricks_client_id="client_id",
            databricks_client_secret="client_secret",
            filters_json=None
        )

        mock_workspace_client.assert_called_with(
            client_id="client_id",
            client_secret="client_secret"
        )

    def test_query_via_databricks_sdk_invalid_query(self):
        """Test SDK query with invalid query parameters."""
        with pytest.raises(ValueError, match="Exactly one of query_text or query_vector must be specified"):
            DatabricksRM._query_via_databricks_sdk(
                index_name="test_index",
                k=3,
                columns=["id", "text"],
                query_type="ANN",
                query_text="test query",
                query_vector=[0.1, 0.2, 0.3],  # Both specified
                databricks_token="test_token",
                databricks_endpoint="https://test.databricks.com",
                databricks_client_id=None,
                databricks_client_secret=None,
                filters_json=None
            )

    @patch('requests.post')
    def test_query_via_requests(self, mock_post):
        """Test querying via requests."""
        mock_response = Mock()
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        result = DatabricksRM._query_via_requests(
            index_name="test_index",
            k=3,
            columns=["id", "text"],
            databricks_token="test_token",
            databricks_endpoint="https://test.databricks.com",
            databricks_client_id=None,
            databricks_client_secret=None,
            query_type="ANN",
            query_text="test query",
            query_vector=None,
            filters_json=None
        )

        assert result == {"result": "success"}
        mock_post.assert_called_once()

    @patch('requests.post')
    @patch('dspy.retrieve.databricks_rm._get_oauth_token')
    def test_query_via_requests_with_oauth(self, mock_get_token, mock_post):
        """Test querying via requests with OAuth authentication."""
        mock_get_token.return_value = "oauth_token"
        mock_response = Mock()
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        DatabricksRM._query_via_requests(
            index_name="test_index",
            k=3,
            columns=["id", "text"],
            databricks_token="old_token",
            databricks_endpoint="https://test.databricks.com",
            databricks_client_id="client_id",
            databricks_client_secret="client_secret",
            query_type="ANN",
            query_text="test query",
            query_vector=None,
            filters_json=None
        )

        mock_get_token.assert_called_once()
        # Verify that the OAuth token was used in the request
        call_args = mock_post.call_args
        headers = call_args[1]['headers']
        assert headers['Authorization'] == 'Bearer oauth_token'

    @patch('requests.post')
    def test_query_via_requests_error_response(self, mock_post):
        """Test requests query with error response."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "error_code": "INVALID_REQUEST",
            "message": "Invalid query parameters"
        }
        mock_post.return_value = mock_response

        with pytest.raises(Exception, match="ERROR: INVALID_REQUEST -- Invalid query parameters"):
            DatabricksRM._query_via_requests(
                index_name="test_index",
                k=3,
                columns=["id", "text"],
                databricks_token="test_token",
                databricks_endpoint="https://test.databricks.com",
                databricks_client_id=None,
                databricks_client_secret=None,
                query_type="ANN",
                query_text="test query",
                query_vector=None,
                filters_json=None
            )

    def test_query_via_requests_invalid_query(self):
        """Test requests query with invalid query parameters."""
        with pytest.raises(ValueError, match="Exactly one of query_text or query_vector must be specified"):
            DatabricksRM._query_via_requests(
                index_name="test_index",
                k=3,
                columns=["id", "text"],
                databricks_token="test_token",
                databricks_endpoint="https://test.databricks.com",
                databricks_client_id=None,
                databricks_client_secret=None,
                query_type="ANN",
                query_text=None,  # Neither specified
                query_vector=None,
                filters_json=None
            )


class TestGetOAuthToken:
    """Test cases for the _get_oauth_token function."""

    @patch('requests.post')
    def test_get_oauth_token_success(self, mock_post):
        """Test successful OAuth token retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = {"access_token": "test_oauth_token"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        token = _get_oauth_token(
            index_name="test_index",
            databricks_endpoint="https://test.databricks.com",
            databricks_client_id="client_id",
            databricks_client_secret="client_secret"
        )

        assert token == "test_oauth_token"
        mock_post.assert_called_once()

        # Verify the request parameters
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://test.databricks.com/oidc/v1/token"
        assert call_args[1]['auth'] == ("client_id", "client_secret")

    @patch('requests.post')
    def test_get_oauth_token_failure(self, mock_post):
        """Test OAuth token retrieval failure."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Unauthorized")
        mock_post.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            _get_oauth_token(
                index_name="test_index",
                databricks_endpoint="https://test.databricks.com",
                databricks_client_id="invalid_client_id",
                databricks_client_secret="invalid_client_secret"
            )


# Integration test fixtures and helpers
@pytest.fixture
def databricks_rm_instance():
    """Create a DatabricksRM instance for testing."""
    return DatabricksRM(
        databricks_index_name="test_index",
        databricks_token="test_token",
        databricks_endpoint="https://test.databricks.com",
        k=3
    )


class TestIntegration:
    """Integration tests for the complete workflow."""

    @patch('dspy.retrieve.databricks_rm._databricks_sdk_installed', False)
    @patch('requests.post')
    def test_end_to_end_text_query(self, mock_post, databricks_rm_instance):
        """Test complete end-to-end text query workflow."""
        # Mock the API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "manifest": {
                "columns": [
                    {"name": "id"},
                    {"name": "text"},
                    {"name": "score"}
                ]
            },
            "result": {
                "data_array": [
                    [1, "First document", 0.95],
                    [2, "Second document", 0.85]
                ]
            }
        }
        mock_post.return_value = mock_response

        # Execute the query
        result = databricks_rm_instance.forward("test query")

        # Verify the results
        assert hasattr(result, 'docs')
        assert len(result.docs) == 2
        assert result.docs[0] == "First document"
        assert result.docs[1] == "Second document"
        assert result.doc_ids == [1, 2]

    @patch('dspy.retrieve.databricks_rm._databricks_sdk_installed', False)
    @patch('requests.post')
    def test_end_to_end_with_filters(self, mock_post, databricks_rm_instance):
        """Test complete workflow with filters."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "manifest": {
                "columns": [
                    {"name": "id"},
                    {"name": "text"},
                    {"name": "score"},
                    {"name": "category"}
                ]
            },
            "result": {
                "data_array": [
                    [1, "Filtered document", 0.95, "news"]
                ]
            }
        }
        mock_post.return_value = mock_response

        result = databricks_rm_instance.forward(
            "test query",
            filters_json='{"category": "news"}'
        )

        assert len(result.docs) == 1
        assert result.docs[0] == "Filtered document"

        # Verify that filters were passed to the API
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['filters_json'] == '{"category": "news"}'


# Parametrized tests for different scenarios
@pytest.mark.parametrize("query_type,expected_type", [
    ("ANN", "ANN"),
    ("HYBRID", "HYBRID"),
    ("text", "ANN"),  # Legacy mapping
    ("vector", "ANN")  # Legacy mapping
])
def test_query_type_mapping(query_type, expected_type):
    """Test query type mapping including legacy types."""
    with patch('dspy.retrieve.databricks_rm._databricks_sdk_installed', False), \
            patch('dspy.retrieve.databricks_rm.DatabricksRM._query_via_requests') as mock_query:
        mock_query.return_value = {
            "manifest": {"columns": [{"name": "id"}, {"name": "text"}, {"name": "score"}]},
            "result": {"data_array": []}
        }

        retriever = DatabricksRM(
            databricks_index_name="test_index",
            databricks_token="test_token",
            databricks_endpoint="https://test.databricks.com"
        )

        retriever.forward("test query", query_type=query_type)

        args, kwargs = mock_query.call_args
        assert kwargs['query_type'] == expected_type


@pytest.mark.parametrize("use_agent_framework,expected_type", [
    (True, list),
    (False, type(Mock()))  # Prediction type
])
def test_return_type_based_on_framework(use_agent_framework, expected_type):
    """Test return type based on agent framework setting."""
    with patch('dspy.retrieve.databricks_rm._databricks_sdk_installed', False), \
            patch('dspy.retrieve.databricks_rm.DatabricksRM._query_via_requests') as mock_query, \
            patch('dspy.primitives.prediction.Prediction') as mock_prediction:

        mock_query.return_value = {
            "manifest": {"columns": [{"name": "id"}, {"name": "text"}, {"name": "score"}]},
            "result": {"data_array": [[1, "test", 0.9]]}
        }

        retriever = DatabricksRM(
            databricks_index_name="test_index",
            databricks_token="test_token",
            databricks_endpoint="https://test.databricks.com",
            use_with_databricks_agent_framework=use_agent_framework
        )

        result = retriever.forward("test query")

        if use_agent_framework:
            assert isinstance(result, list)
        else:
            # For non-agent framework, it should create a Prediction object
            mock_prediction.assert_called_once()