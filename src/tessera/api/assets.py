"""Assets API endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tessera.db import AssetDB, ContractDB, ProposalDB, RegistrationDB, TeamDB, get_session
from tessera.models import Asset, AssetCreate, AssetUpdate, Contract, ContractCreate, Proposal
from tessera.models.enums import ChangeType, ContractStatus, RegistrationStatus
from tessera.services import (
    check_compatibility,
    diff_schemas,
    log_contract_published,
    log_proposal_created,
    validate_json_schema,
)

router = APIRouter()


@router.post("", response_model=Asset, status_code=201)
async def create_asset(
    asset: AssetCreate,
    session: AsyncSession = Depends(get_session),
) -> AssetDB:
    """Create a new asset."""
    # Validate owner team exists
    result = await session.execute(select(TeamDB).where(TeamDB.id == asset.owner_team_id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Owner team not found")

    db_asset = AssetDB(
        fqn=asset.fqn,
        owner_team_id=asset.owner_team_id,
        metadata_=asset.metadata,
    )
    session.add(db_asset)
    await session.flush()
    await session.refresh(db_asset)
    return db_asset


@router.get("", response_model=list[Asset])
async def list_assets(
    owner: UUID | None = Query(None, description="Filter by owner team ID"),
    session: AsyncSession = Depends(get_session),
) -> list[AssetDB]:
    """List all assets, optionally filtered by owner."""
    query = select(AssetDB)
    if owner:
        query = query.where(AssetDB.owner_team_id == owner)
    result = await session.execute(query)
    return list(result.scalars().all())


@router.get("/{asset_id}", response_model=Asset)
async def get_asset(
    asset_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> AssetDB:
    """Get an asset by ID."""
    result = await session.execute(select(AssetDB).where(AssetDB.id == asset_id))
    asset = result.scalar_one_or_none()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset


@router.patch("/{asset_id}", response_model=Asset)
async def update_asset(
    asset_id: UUID,
    update: AssetUpdate,
    session: AsyncSession = Depends(get_session),
) -> AssetDB:
    """Update an asset."""
    result = await session.execute(select(AssetDB).where(AssetDB.id == asset_id))
    asset = result.scalar_one_or_none()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    if update.fqn is not None:
        asset.fqn = update.fqn
    if update.owner_team_id is not None:
        asset.owner_team_id = update.owner_team_id
    if update.metadata is not None:
        asset.metadata_ = update.metadata

    await session.flush()
    await session.refresh(asset)
    return asset


@router.post("/{asset_id}/contracts", status_code=201)
async def create_contract(
    asset_id: UUID,
    contract: ContractCreate,
    published_by: UUID = Query(..., description="Team ID of the publisher"),
    force: bool = Query(False, description="Force publish even if breaking (creates audit trail)"),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Publish a new contract for an asset.

    Behavior:
    - If no active contract exists: auto-publish (first contract)
    - If change is compatible: auto-publish, deprecate old contract
    - If change is breaking: create a Proposal for consumer acknowledgment
    - If force=True: publish anyway but log the override

    Returns either a Contract (if published) or a Proposal (if breaking).
    """
    # Verify asset exists
    result = await session.execute(select(AssetDB).where(AssetDB.id == asset_id))
    asset = result.scalar_one_or_none()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    # Verify publisher team exists
    result = await session.execute(select(TeamDB).where(TeamDB.id == published_by))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Publisher team not found")

    # Validate schema is valid JSON Schema
    is_valid, errors = validate_json_schema(contract.schema_def)
    if not is_valid:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_SCHEMA",
                "message": "Invalid JSON Schema",
                "errors": errors,
            },
        )

    # Get current active contract
    result = await session.execute(
        select(ContractDB)
        .where(ContractDB.asset_id == asset_id)
        .where(ContractDB.status == ContractStatus.ACTIVE)
        .order_by(ContractDB.published_at.desc())
        .limit(1)
    )
    current_contract = result.scalar_one_or_none()

    # Helper to create and return the new contract
    async def publish_contract() -> ContractDB:
        db_contract = ContractDB(
            asset_id=asset_id,
            version=contract.version,
            schema_def=contract.schema_def,
            compatibility_mode=contract.compatibility_mode,
            guarantees=contract.guarantees.model_dump() if contract.guarantees else None,
            published_by=published_by,
        )
        session.add(db_contract)

        # Deprecate old contract if exists
        if current_contract:
            current_contract.status = ContractStatus.DEPRECATED

        await session.flush()
        await session.refresh(db_contract)
        return db_contract

    # No existing contract = first publish, auto-approve
    if not current_contract:
        new_contract = await publish_contract()
        await log_contract_published(
            session=session,
            contract_id=new_contract.id,
            publisher_id=published_by,
            version=new_contract.version,
        )
        return {
            "action": "published",
            "contract": Contract.model_validate(new_contract).model_dump(),
        }

    # Diff schemas and check compatibility
    is_compatible, breaking_changes = check_compatibility(
        current_contract.schema_def,
        contract.schema_def,
        current_contract.compatibility_mode,
    )
    diff_result = diff_schemas(current_contract.schema_def, contract.schema_def)

    # Compatible change = auto-publish
    if is_compatible:
        new_contract = await publish_contract()
        await log_contract_published(
            session=session,
            contract_id=new_contract.id,
            publisher_id=published_by,
            version=new_contract.version,
            change_type=str(diff_result.change_type),
        )
        return {
            "action": "published",
            "change_type": str(diff_result.change_type),
            "contract": Contract.model_validate(new_contract).model_dump(),
        }

    # Breaking change with force flag = publish anyway (logged)
    if force:
        new_contract = await publish_contract()
        await log_contract_published(
            session=session,
            contract_id=new_contract.id,
            publisher_id=published_by,
            version=new_contract.version,
            change_type=str(diff_result.change_type),
            force=True,
        )
        return {
            "action": "force_published",
            "change_type": str(diff_result.change_type),
            "breaking_changes": [bc.to_dict() for bc in breaking_changes],
            "contract": Contract.model_validate(new_contract).model_dump(),
            "warning": "Breaking change was force-published. Consumers may be affected.",
        }

    # Breaking change without force = create proposal
    db_proposal = ProposalDB(
        asset_id=asset_id,
        proposed_schema=contract.schema_def,
        change_type=diff_result.change_type,
        breaking_changes=[bc.to_dict() for bc in breaking_changes],
        proposed_by=published_by,
    )
    session.add(db_proposal)
    await session.flush()
    await session.refresh(db_proposal)

    await log_proposal_created(
        session=session,
        proposal_id=db_proposal.id,
        asset_id=asset_id,
        proposer_id=published_by,
        change_type=str(diff_result.change_type),
        breaking_changes=[bc.to_dict() for bc in breaking_changes],
    )

    # TODO: Notify consumers via webhook

    return {
        "action": "proposal_created",
        "change_type": str(diff_result.change_type),
        "breaking_changes": [bc.to_dict() for bc in breaking_changes],
        "proposal": Proposal.model_validate(db_proposal).model_dump(),
        "message": "Breaking change detected. Proposal created for consumer acknowledgment.",
    }


@router.get("/{asset_id}/contracts", response_model=list[Contract])
async def list_asset_contracts(
    asset_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> list[ContractDB]:
    """List all contracts for an asset."""
    result = await session.execute(
        select(ContractDB).where(ContractDB.asset_id == asset_id).order_by(ContractDB.published_at)
    )
    return list(result.scalars().all())


@router.post("/{asset_id}/impact")
async def analyze_impact(
    asset_id: UUID,
    proposed_schema: dict[str, Any],
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Analyze the impact of a proposed schema change.

    Compares the proposed schema against the current active contract
    and identifies breaking changes and impacted consumers.
    """
    # Validate proposed schema is valid JSON Schema
    is_valid, errors = validate_json_schema(proposed_schema)
    if not is_valid:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_SCHEMA",
                "message": "Invalid JSON Schema",
                "errors": errors,
            },
        )

    # Verify asset exists
    result = await session.execute(select(AssetDB).where(AssetDB.id == asset_id))
    asset = result.scalar_one_or_none()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    # Get the current active contract
    result = await session.execute(
        select(ContractDB)
        .where(ContractDB.asset_id == asset_id)
        .where(ContractDB.status == ContractStatus.ACTIVE)
        .order_by(ContractDB.published_at.desc())
        .limit(1)
    )
    current_contract = result.scalar_one_or_none()

    # No active contract = safe to publish (first contract)
    if not current_contract:
        return {
            "change_type": "minor",
            "breaking_changes": [],
            "impacted_consumers": [],
            "safe_to_publish": True,
        }

    # Diff the schemas
    diff_result = diff_schemas(current_contract.schema_def, proposed_schema)
    breaking = diff_result.breaking_for_mode(current_contract.compatibility_mode)

    # Get impacted consumers (registrations for this contract)
    result = await session.execute(
        select(RegistrationDB)
        .where(RegistrationDB.contract_id == current_contract.id)
        .where(RegistrationDB.status == RegistrationStatus.ACTIVE)
    )
    registrations = result.scalars().all()

    # Get team names for impacted consumers
    impacted_consumers = []
    for reg in registrations:
        team_result = await session.execute(
            select(TeamDB).where(TeamDB.id == reg.consumer_team_id)
        )
        team = team_result.scalar_one_or_none()
        impacted_consumers.append({
            "team_id": str(reg.consumer_team_id),
            "team_name": team.name if team else "Unknown",
            "status": str(reg.status),
            "pinned_version": reg.pinned_version,
        })

    return {
        "change_type": str(diff_result.change_type),
        "breaking_changes": [bc.to_dict() for bc in breaking],
        "impacted_consumers": impacted_consumers,
        "safe_to_publish": len(breaking) == 0,
    }


@router.get("/{asset_id}/lineage")
async def get_lineage(
    asset_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Get the dependency lineage for an asset.

    Returns teams that consume this asset (downstream).
    Upstream tracking requires additional modeling (not yet implemented).
    """
    # Verify asset exists
    result = await session.execute(select(AssetDB).where(AssetDB.id == asset_id))
    asset = result.scalar_one_or_none()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    # Get all contracts for this asset
    contracts_result = await session.execute(
        select(ContractDB).where(ContractDB.asset_id == asset_id)
    )
    contracts = contracts_result.scalars().all()
    contract_ids = [c.id for c in contracts]

    # Get all registrations (consumers) for these contracts
    downstream = []
    if contract_ids:
        regs_result = await session.execute(
            select(RegistrationDB).where(RegistrationDB.contract_id.in_(contract_ids))
        )
        registrations = regs_result.scalars().all()

        # Get unique consumer teams
        consumer_team_ids = set(reg.consumer_team_id for reg in registrations)
        for team_id in consumer_team_ids:
            team_result = await session.execute(select(TeamDB).where(TeamDB.id == team_id))
            team = team_result.scalar_one_or_none()
            if team:
                # Get the registrations for this team
                team_regs = [r for r in registrations if r.consumer_team_id == team_id]
                downstream.append({
                    "team_id": str(team_id),
                    "team_name": team.name,
                    "registrations": [
                        {
                            "contract_id": str(r.contract_id),
                            "status": str(r.status),
                            "pinned_version": r.pinned_version,
                        }
                        for r in team_regs
                    ],
                })

    return {
        "asset_id": str(asset_id),
        "asset_fqn": asset.fqn,
        "owner_team_id": str(asset.owner_team_id),
        "upstream": [],  # Not yet modeled - would require asset-to-asset dependencies
        "downstream": downstream,
    }
