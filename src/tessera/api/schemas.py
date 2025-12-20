"""Schemas API endpoints."""

from typing import Any

from fastapi import APIRouter

from tessera.services import check_schema_validity

router = APIRouter()


@router.post("/validate")
async def validate_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Validate a JSON Schema.

    Checks whether the provided dictionary is a valid JSON Schema (Draft 7).

    Returns:
        - valid: boolean indicating if the schema is valid
        - errors: list of error messages (empty if valid)
    """
    return check_schema_validity(schema)
