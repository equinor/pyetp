import dataclasses
import re

import pytest

import energistics.uris
from energistics.uris import DataObjectURI, DataspaceURI


def test_dataspace_uri_pattern() -> None:
    invalid_uris = [
        "a",
        "eml://",
        "eml:///dataspace",
        "eml:///dataspace(foo/bar)",
        "aml:///dataspace('foo/bar')",
    ]

    for iu in invalid_uris:
        m = re.match(energistics.uris.DATASPACE_URI_PATTERN, iu)
        assert m is None
        with pytest.raises(ValueError):
            DataspaceURI.from_uri(iu)

    paths = [
        None,
        "",
        "foo/bar",
        "/foo",
        "foo/bar/baz",
        "/foo/bar/baz",
    ]
    valid_uris = [
        *["eml:///" + (f"dataspace('{p}')" if p is not None else "") for p in paths],
    ]

    for p, vu in zip(paths, valid_uris):
        m = re.match(energistics.uris.DATASPACE_URI_PATTERN, vu)
        assert m is not None
        assert m.group(1) == p
        dataspace_uri = DataspaceURI.from_uri(vu)
        assert dataspace_uri.uri == vu
        assert dataspace_uri.dataspace == p
        dataspace_uri2 = DataspaceURI(**dataclasses.asdict(dataspace_uri))
        assert dataspace_uri == dataspace_uri2
        dataspace_uri3 = DataspaceURI.from_uri(dataspace_uri)
        assert dataspace_uri3 == dataspace_uri

        dataspace_uri4 = DataspaceURI.from_dataspace_path(dataspace_uri.dataspace)
        assert dataspace_uri == dataspace_uri4


def test_data_object_uri_pattern() -> None:
    paths = [
        None,
        "",
        "foo/bar",
        "/foo",
        "foo/bar/baz",
        "/foo/bar/baz",
    ]
    invalid_uris = [
        "a",
        "eml://",
        "eml:///dataspace",
        "eml:///dataspace(foo/bar)",
        "aml:///dataspace('foo/bar')",
        *["eml:///" + (f"dataspace('{p}')" if p is not None else "") for p in paths],
    ]

    for iu in invalid_uris:
        m = re.match(energistics.uris.DATA_OBJECT_URI_PATTERN, iu)
        assert m is None
        with pytest.raises(ValueError):
            DataObjectURI.from_uri(iu)

    valid_uri = "eml:///witsml20.Well(ec8c3f16-1454-4f36-ae10-27d2a2680cf2)"

    m = re.match(energistics.uris.DATA_OBJECT_URI_PATTERN, valid_uri)

    assert m is not None
    assert m.group("dataspace") is None
    assert m.group("domain") == "witsml"
    assert m.group("domain_version") == "20"
    assert m.group("object_type") == "Well"
    assert m.group("uuid") == "ec8c3f16-1454-4f36-ae10-27d2a2680cf2"
    assert m.group("uuid2") is None
    assert m.group("version") is None

    valid_uris = [
        "eml:///witsml20.Well(ec8c3f16-1454-4f36-ae10-27d2a2680cf2)",
        "eml:///witsml20.Well(uuid=ec8c3f16-1454-4f36-ae10-27d2a2680cf2,version='1.0')",
        "eml:///dataspace('/folder-name/project-name')/resqml20.obj_HorizonInterpretation(uuid=421a7a05-033a-450d-bcef-051352023578,version='2.0')",
        "eml:///dataspace('test/bar')/eml20.EpcExternalPartReference(12345678-abcd-a1b2-c3d4-ab12cd34ef56)",
    ]

    groups: dict[str, list[str | None]] = {
        "dataspace": [
            None,
            None,
            "/folder-name/project-name",
            "test/bar",
        ],
        "domain": [
            "witsml",
            "witsml",
            "resqml",
            "eml",
        ],
        "domain_version": [
            "20",
            "20",
            "20",
            "20",
        ],
        "object_type": [
            "Well",
            "Well",
            "obj_HorizonInterpretation",
            "EpcExternalPartReference",
        ],
        "uuid": [
            "ec8c3f16-1454-4f36-ae10-27d2a2680cf2",
            None,
            None,
            "12345678-abcd-a1b2-c3d4-ab12cd34ef56",
        ],
        "uuid2": [
            None,
            "ec8c3f16-1454-4f36-ae10-27d2a2680cf2",
            "421a7a05-033a-450d-bcef-051352023578",
            None,
        ],
        "version": [None, "1.0", "2.0", None],
    }

    for i, vu in enumerate(valid_uris):
        m = re.match(energistics.uris.DATA_OBJECT_URI_PATTERN, vu)
        assert m is not None
        for k, v in groups.items():
            if m.group(k) is None:
                assert v[i] is None
            else:
                assert m.group(k) == v[i]

        data_object_uri = DataObjectURI.from_uri(vu)

        assert data_object_uri.uri == vu
        assert data_object_uri.dataspace == groups["dataspace"][i]
        assert data_object_uri.domain == groups["domain"][i]
        assert data_object_uri.domain_version == groups["domain_version"][i]
        assert data_object_uri.object_type == groups["object_type"][i]
        assert (
            data_object_uri.uuid == groups["uuid"][i]
            or data_object_uri.uuid == groups["uuid2"][i]
        )
        assert data_object_uri.version == groups["version"][i]
        data_object_uri2 = DataObjectURI(**dataclasses.asdict(data_object_uri))
        assert data_object_uri == data_object_uri2
        data_object_uri3 = DataObjectURI.from_uri(data_object_uri)
        assert data_object_uri == data_object_uri3
        data_object_uri4 = DataObjectURI.from_parts(
            dataspace=data_object_uri.dataspace,
            domain=data_object_uri.domain,
            domain_version=data_object_uri.domain_version,
            object_type=data_object_uri.object_type,
            uuid=data_object_uri.uuid,
            version=data_object_uri.version,
        )
        assert data_object_uri4 == data_object_uri
