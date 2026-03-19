import uuid

import energistics.base
from energistics.etp.v12.datatypes.object import (
    ContextInfo,
    DeletedResource,
    Edge,
    Resource,
)
from energistics.etp.v12.protocol.discovery import (
    GetDeletedResources,
    GetDeletedResourcesResponse,
    GetResources,
    GetResourcesEdgesResponse,
    GetResourcesResponse,
)
from tests.test_etp_objects.conftest import avro_roundtrip


def test_get_resources() -> None:
    gr = GetResources(
        context=ContextInfo(
            uri="eml:///dataspace('foo/bar')",
            depth=1,
            data_object_types=["resqml20.*", "eml20.*"],
            navigable_edges="Primary",
        ),
        scope="sources",
        store_last_write_filter=23,
        active_status_filter="Active",
    )

    assert gr._protocol == energistics.base.Protocol.DISCOVERY
    assert gr._message_type == 1
    assert not gr._is_multipart

    ret_gr = avro_roundtrip(gr)
    assert gr == ret_gr


def test_get_resources_response() -> None:
    grr = GetResourcesResponse(
        resources=[
            Resource(
                uri=f"eml:///dataspace('foo/bar')/witsml20.Well({uuid.uuid4()!s})",
                name="Welly",
                last_changed=10,
                store_last_write=2000,
                store_created=1500,
                active_status="Inactive",
            ),
        ],
    )

    assert grr._protocol == energistics.base.Protocol.DISCOVERY
    assert grr._message_type == 4
    assert grr._is_multipart

    ret_grr = avro_roundtrip(grr)
    assert grr == ret_grr


def test_get_resources_edges_response() -> None:
    grer = GetResourcesEdgesResponse(
        edges=[
            Edge(
                source_uri=f"eml:///dataspace('foo/bar')/resqml20.Grid2dRepresentation({uuid.uuid4()!s})",
                target_uri=f"eml:///dataspace('foo/bar')/resqml20.Grid2dRepresentation({uuid.uuid4()!s})",
                relationship_kind="Secondary",
            ),
        ],
    )

    assert grer._protocol == energistics.base.Protocol.DISCOVERY
    assert grer._message_type == 7
    assert grer._is_multipart

    ret_grer = avro_roundtrip(grer)
    assert grer == ret_grer


def test_get_deleted_resources() -> None:
    gdr = GetDeletedResources(
        dataspace_uri="eml:///dataspace('foo/bar')",
        delete_time_filter=10,
        data_object_types=[
            "witsml21.*",
        ],
    )

    assert gdr._protocol == energistics.base.Protocol.DISCOVERY
    assert gdr._message_type == 5
    assert not gdr._is_multipart

    ret_gdr = avro_roundtrip(gdr)
    assert gdr == ret_gdr


def test_get_deleted_resources_response() -> None:
    gdrr = GetDeletedResourcesResponse(
        deleted_resources=[
            DeletedResource(
                uri=f"eml:///dataspace('foo/bar')/witsml21.Well({uuid.uuid4()!s})",
                deleted_time=12355,
            ),
        ],
    )

    assert gdrr._protocol == energistics.base.Protocol.DISCOVERY
    assert gdrr._message_type == 6
    assert gdrr._is_multipart

    ret_gdrr = avro_roundtrip(gdrr)
    assert gdrr == ret_gdrr
