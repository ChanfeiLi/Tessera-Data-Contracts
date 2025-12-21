"""Webhook delivery service."""

import asyncio
import hashlib
import hmac
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx

from tessera.config import settings
from tessera.models.webhook import (
    AcknowledgmentPayload,
    BreakingChange,
    ContractPublishedPayload,
    ImpactedConsumer,
    ProposalCreatedPayload,
    ProposalStatusPayload,
    WebhookEvent,
    WebhookEventType,
)

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAYS = [1, 5, 30]  # seconds between retries


def _sign_payload(payload: str, secret: str) -> str:
    """Sign a payload with HMAC-SHA256."""
    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


async def _deliver_webhook(event: WebhookEvent) -> bool:
    """Deliver a webhook event to the configured URL.

    Returns True if delivery succeeded, False otherwise.
    """
    if not settings.webhook_url:
        logger.debug("No webhook URL configured, skipping delivery")
        return True

    payload = event.model_dump_json()
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "X-Tessera-Event": event.event.value,
        "X-Tessera-Timestamp": event.timestamp.isoformat(),
    }

    if settings.webhook_secret:
        signature = _sign_payload(payload, settings.webhook_secret)
        headers["X-Tessera-Signature"] = f"sha256={signature}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt, delay in enumerate(RETRY_DELAYS):
            try:
                response = await client.post(
                    settings.webhook_url,
                    content=payload,
                    headers=headers,
                )
                if response.status_code < 300:
                    logger.info(
                        "Webhook delivered: %s to %s",
                        event.event.value,
                        settings.webhook_url,
                    )
                    return True
                logger.warning(
                    "Webhook delivery failed (attempt %d): %s %s",
                    attempt + 1,
                    response.status_code,
                    response.text[:200],
                )
            except httpx.RequestError as e:
                logger.warning(
                    "Webhook delivery error (attempt %d): %s",
                    attempt + 1,
                    str(e),
                )

            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(delay)

    logger.error(
        "Webhook delivery failed after %d attempts: %s",
        MAX_RETRIES,
        event.event.value,
    )
    return False


def _fire_and_forget(event: WebhookEvent) -> None:
    """Schedule webhook delivery without blocking."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_deliver_webhook(event))
    except RuntimeError:
        # No running loop - skip webhook (happens in tests without async context)
        logger.debug("No event loop, skipping webhook: %s", event.event.value)


async def send_proposal_created(
    proposal_id: UUID,
    asset_id: UUID,
    asset_fqn: str,
    producer_team_id: UUID,
    producer_team_name: str,
    proposed_version: str,
    breaking_changes: list[dict[str, Any]],
    impacted_consumers: list[dict[str, Any]],
) -> None:
    """Send webhook when a breaking change proposal is created."""
    event = WebhookEvent(
        event=WebhookEventType.PROPOSAL_CREATED,
        timestamp=datetime.now(UTC),
        payload=ProposalCreatedPayload(
            proposal_id=proposal_id,
            asset_id=asset_id,
            asset_fqn=asset_fqn,
            producer_team_id=producer_team_id,
            producer_team_name=producer_team_name,
            proposed_version=proposed_version,
            breaking_changes=[
                BreakingChange(
                    change_type=c.get("change_type", "unknown"),
                    path=c.get("path", ""),
                    message=c.get("message", ""),
                    details=c.get("details"),
                )
                for c in breaking_changes
            ],
            impacted_consumers=[
                ImpactedConsumer(
                    team_id=c["team_id"],
                    team_name=c["team_name"],
                    pinned_version=c.get("pinned_version"),
                )
                for c in impacted_consumers
            ],
        ),
    )
    _fire_and_forget(event)


async def send_proposal_acknowledged(
    proposal_id: UUID,
    asset_id: UUID,
    asset_fqn: str,
    consumer_team_id: UUID,
    consumer_team_name: str,
    response: str,
    migration_deadline: datetime | None,
    notes: str | None,
    pending_count: int,
    acknowledged_count: int,
) -> None:
    """Send webhook when a consumer acknowledges a proposal."""
    event = WebhookEvent(
        event=WebhookEventType.PROPOSAL_ACKNOWLEDGED,
        timestamp=datetime.now(UTC),
        payload=AcknowledgmentPayload(
            proposal_id=proposal_id,
            asset_id=asset_id,
            asset_fqn=asset_fqn,
            consumer_team_id=consumer_team_id,
            consumer_team_name=consumer_team_name,
            response=response,
            migration_deadline=migration_deadline,
            notes=notes,
            pending_count=pending_count,
            acknowledged_count=acknowledged_count,
        ),
    )
    _fire_and_forget(event)


async def send_proposal_status_change(
    event_type: WebhookEventType,
    proposal_id: UUID,
    asset_id: UUID,
    asset_fqn: str,
    status: str,
    actor_team_id: UUID | None = None,
    actor_team_name: str | None = None,
) -> None:
    """Send webhook when proposal status changes (approved, rejected, etc.)."""
    event = WebhookEvent(
        event=event_type,
        timestamp=datetime.now(UTC),
        payload=ProposalStatusPayload(
            proposal_id=proposal_id,
            asset_id=asset_id,
            asset_fqn=asset_fqn,
            status=status,
            actor_team_id=actor_team_id,
            actor_team_name=actor_team_name,
        ),
    )
    _fire_and_forget(event)


async def send_contract_published(
    contract_id: UUID,
    asset_id: UUID,
    asset_fqn: str,
    version: str,
    producer_team_id: UUID,
    producer_team_name: str,
    from_proposal_id: UUID | None = None,
) -> None:
    """Send webhook when a contract is published."""
    event = WebhookEvent(
        event=WebhookEventType.CONTRACT_PUBLISHED,
        timestamp=datetime.now(UTC),
        payload=ContractPublishedPayload(
            contract_id=contract_id,
            asset_id=asset_id,
            asset_fqn=asset_fqn,
            version=version,
            producer_team_id=producer_team_id,
            producer_team_name=producer_team_name,
            from_proposal_id=from_proposal_id,
        ),
    )
    _fire_and_forget(event)
