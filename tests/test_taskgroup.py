import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import websockets

from energistics.etp.v12.datatypes.object import (
    ActiveStatusKind,
    ContextScopeKind,
    Edge,
    RelationshipKind,
    Resource,
)
from energistics.etp.v12.protocol.dataspace import GetDataspaces
from energistics.etp.v12.protocol.discovery import (
    GetResourcesEdgesResponse,
    GetResourcesResponse,
)
from pyetp.client import ETPClient
from rddms_io.client import RDDMSClient
from rddms_io.data_types import RDDMSModel


@pytest.mark.asyncio
async def test_etp_client_send_multipart_taskgroup_success() -> None:
    ws = MagicMock()
    ws.protocol.max_message_size = 100_000
    ws.recv = AsyncMock(side_effect=websockets.ConnectionClosedOK(rcvd=None, sent=None))

    sent_messages: list[bytes] = []

    async def mock_send(msg: bytes) -> None:
        await asyncio.sleep(0.01)
        sent_messages.append(msg)

    ws.send = AsyncMock(side_effect=mock_send)

    client = ETPClient(ws=ws, use_compression=False)
    client.negotiated_compression = None
    client._recv_events = {}

    try:
        main_msg = GetDataspaces()
        part1 = GetDataspaces()
        part2 = GetDataspaces()

        msg_id = await client.send(main_msg, multi_part_bodies=[part1, part2])

        assert msg_id is not None
        assert len(sent_messages) == 3
        assert ws.send.await_count == 3
    finally:
        client._recvtask.cancel()
        try:
            await client._recvtask
        except (asyncio.CancelledError, websockets.ConnectionClosedOK):
            pass


@pytest.mark.asyncio
async def test_etp_client_send_multipart_taskgroup_cancellation_on_failure() -> None:
    ws = MagicMock()
    ws.protocol.max_message_size = 100_000
    ws.recv = AsyncMock(side_effect=websockets.ConnectionClosedOK(rcvd=None, sent=None))

    cancelled_count = 0
    call_count = 0

    async def failing_send(msg: bytes) -> None:
        nonlocal cancelled_count, call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("Simulated websocket transport failure")
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            cancelled_count += 1
            raise

    ws.send = AsyncMock(side_effect=failing_send)

    client = ETPClient(ws=ws, use_compression=False)
    client.negotiated_compression = None
    client._recv_events = {}

    try:
        main_msg = GetDataspaces()
        part1 = GetDataspaces()
        part2 = GetDataspaces()

        with pytest.raises(ExceptionGroup) as exc_info:
            await client.send(main_msg, multi_part_bodies=[part1, part2])

        assert any(
            isinstance(e, RuntimeError)
            and "Simulated websocket transport failure" in str(e)
            for e in exc_info.value.exceptions
        )
        assert cancelled_count > 0
    finally:
        client._recvtask.cancel()
        try:
            await client._recvtask
        except (asyncio.CancelledError, websockets.ConnectionClosedOK):
            pass


@pytest.mark.asyncio
async def test_rddms_client_list_linked_objects_taskgroup_success() -> None:
    etp_client = MagicMock()

    u1 = "eml:///dataspace('demo')/resqml20.obj_LocalDepth3dCrs(11111111-1111-1111-1111-111111111111)"
    u2 = "eml:///dataspace('demo')/resqml20.obj_Grid2dRepresentation(22222222-2222-2222-2222-222222222222)"

    edge_source = Edge(
        source_uri=u1,
        target_uri=u2,
        relationship_kind=RelationshipKind.PRIMARY,
    )
    edge_target = Edge(
        source_uri=u2,
        target_uri=u1,
        relationship_kind=RelationshipKind.PRIMARY,
    )
    res_source = Resource(
        uri=u1,
        name="Crs",
        last_changed=0,
        store_last_write=0,
        store_created=0,
        active_status=ActiveStatusKind.ACTIVE,
    )

    async def mock_send_and_recv(msg: object) -> list[object]:
        await asyncio.sleep(0.01)
        if getattr(msg, "scope", None) == ContextScopeKind.TARGETS_OR_SELF:
            return [GetResourcesEdgesResponse(edges=[edge_target])]
        return [
            GetResourcesEdgesResponse(edges=[edge_source]),
            GetResourcesResponse(resources=[res_source]),
        ]

    etp_client.send_and_recv = AsyncMock(side_effect=mock_send_and_recv)

    rddms = RDDMSClient(etp_client=etp_client)

    linked = await rddms.list_linked_objects(u1)

    assert len(linked.source_edges) == 1
    assert len(linked.target_edges) == 1
    assert edge_source in linked.source_edges
    assert edge_target in linked.target_edges
    assert etp_client.send_and_recv.await_count == 2


@pytest.mark.asyncio
async def test_rddms_client_list_linked_objects_taskgroup_cancellation_on_failure() -> (
    None
):
    etp_client = MagicMock()
    cancelled_count = 0

    async def mock_send_and_recv(msg: object) -> list[object]:
        nonlocal cancelled_count
        if getattr(msg, "scope", None) == ContextScopeKind.TARGETS_OR_SELF:
            raise RuntimeError("Targets query failed")
        try:
            await asyncio.sleep(1.0)
            return []
        except asyncio.CancelledError:
            cancelled_count += 1
            raise

    etp_client.send_and_recv = AsyncMock(side_effect=mock_send_and_recv)

    rddms = RDDMSClient(etp_client=etp_client)

    u1 = "eml:///dataspace('demo')/resqml20.obj_Grid2dRepresentation(22222222-2222-2222-2222-222222222222)"
    with pytest.raises(ExceptionGroup) as exc_info:
        await rddms.list_linked_objects(u1)

    assert any(
        isinstance(e, RuntimeError) and "Targets query failed" in str(e)
        for e in exc_info.value.exceptions
    )
    assert cancelled_count > 0


@pytest.mark.asyncio
async def test_rddms_client_download_models_taskgroup_success() -> None:
    etp_client = MagicMock()
    rddms = RDDMSClient(etp_client=etp_client)

    m1 = RDDMSModel(obj=MagicMock(), arrays={}, linked_models=[])
    m2 = RDDMSModel(obj=MagicMock(), arrays={}, linked_models=[])

    async def mock_download_model(
        ml_uri: str, download_arrays: bool, download_linked_objects: bool
    ) -> RDDMSModel:
        await asyncio.sleep(0.01)
        if "1111" in ml_uri:
            return m1
        return m2

    rddms._download_model = AsyncMock(side_effect=mock_download_model)  # type: ignore[method-assign]

    uris = [
        "eml:///dataspace('demo')/resqml20.obj_LocalDepth3dCrs(11111111-1111-1111-1111-111111111111)",
        "eml:///dataspace('demo')/resqml20.obj_LocalDepth3dCrs(22222222-2222-2222-2222-222222222222)",
    ]
    models = await rddms.download_models(uris)

    assert len(models) == 2
    assert models[0] == m1
    assert models[1] == m2
    assert rddms._download_model.await_count == 2
