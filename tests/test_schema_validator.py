"""Tests for JSON Schema validation service."""

import pytest

from tessera.services.schema_validator import (
    SchemaValidationError,
    check_schema_validity,
    validate_json_schema,
    validate_schema_or_raise,
)


class TestValidateJsonSchema:
    """Test validate_json_schema function."""

    def test_valid_schema(self):
        """A valid JSON Schema should pass validation."""
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"},
            },
            "required": ["id"],
        }
        is_valid, errors = validate_json_schema(schema)
        assert is_valid
        assert errors == []

    def test_valid_complex_schema(self):
        """A complex valid schema with nested properties should pass."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "email": {"type": "string", "format": "email"},
                    },
                    "required": ["id"],
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
        is_valid, errors = validate_json_schema(schema)
        assert is_valid
        assert errors == []

    def test_valid_schema_with_enum(self):
        """Schema with enum should be valid."""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive", "pending"]},
            },
        }
        is_valid, errors = validate_json_schema(schema)
        assert is_valid
        assert errors == []

    def test_invalid_type_value(self):
        """Schema with invalid type value should fail."""
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "not_a_type"},  # Invalid type
            },
        }
        is_valid, errors = validate_json_schema(schema)
        assert not is_valid
        assert len(errors) > 0
        assert any("not_a_type" in e for e in errors)

    def test_non_dict_schema(self):
        """Non-dict schema should fail."""
        is_valid, errors = validate_json_schema("not a dict")  # type: ignore
        assert not is_valid
        assert "must be an object" in errors[0]

    def test_invalid_required_not_array(self):
        """Required must be an array."""
        schema = {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": "id",  # Should be array
        }
        is_valid, errors = validate_json_schema(schema)
        assert not is_valid
        assert len(errors) > 0


class TestValidateSchemaOrRaise:
    """Test validate_schema_or_raise function."""

    def test_valid_schema_no_error(self):
        """Valid schema should not raise."""
        schema = {"type": "object", "properties": {"id": {"type": "integer"}}}
        validate_schema_or_raise(schema)  # Should not raise

    def test_invalid_schema_raises(self):
        """Invalid schema should raise SchemaValidationError."""
        schema = {"type": "object", "properties": {"id": {"type": "invalid_type"}}}
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema_or_raise(schema)
        assert "Invalid JSON Schema" in str(exc_info.value.message)


class TestCheckSchemaValidity:
    """Test check_schema_validity function."""

    def test_valid_schema(self):
        """Valid schema should return valid=True."""
        schema = {"type": "object", "properties": {"id": {"type": "integer"}}}
        result = check_schema_validity(schema)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_invalid_schema(self):
        """Invalid schema should return valid=False with errors."""
        schema = {"type": "object", "properties": {"id": {"type": "bad_type"}}}
        result = check_schema_validity(schema)
        assert result["valid"] is False
        assert len(result["errors"]) > 0
