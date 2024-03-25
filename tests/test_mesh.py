import numpy as np
import pytest
import resqpy.model as rq
import resqpy.property as rqp
import resqpy.unstructured as rug

from pyetp.client import MAXPAYLOADSIZE, ETPClient, ETPError
from pyetp.uri import DataspaceURI
from pyetp.resqml_objects import ContinuousProperty, DiscreteProperty


@pytest.mark.parametrize('input_mesh_file', ['./data/model_hexa_0.epc'])
@pytest.mark.asyncio
async def test_mesh(eclient: ETPClient, duri: DataspaceURI, input_mesh_file: str):
    model = rq.Model(input_mesh_file)
    assert model is not None

    hexa_uuids = model.uuids(obj_type='UnstructuredGridRepresentation')
    hexa = rug.HexaGrid(model, uuid=hexa_uuids[0])
    assert hexa is not None
    assert hexa.nodes_per_face is not None, "hexamesh object is incomplete"
    assert hexa.nodes_per_face_cl is not None, "hexamesh object is incomplete"
    assert hexa.faces_per_cell is not None, "hexamesh object is incomplete"
    assert hexa.faces_per_cell_cl is not None, "hexamesh object is incomplete"
    assert hexa.cell_face_is_right_handed is not None, "hexamesh object is incomplete"

    uuids = model.uuids(obj_type='ContinuousProperty')
    prop_titles = [rqp.Property(model, uuid=u).title for u in uuids]
    uuids = model.uuids(obj_type='DiscreteProperty')
    prop_titles = prop_titles + [rqp.Property(model, uuid=u).title for u in uuids]

    rddms_uris, prop_uris = await eclient.put_epc_mesh(input_mesh_file, hexa.title, prop_titles, 23031, duri)
    uns, points, nodes_per_face, nodes_per_face_cl, faces_per_cell, faces_per_cell_cl, cell_face_is_right_handed = await eclient.get_epc_mesh(rddms_uris[0], rddms_uris[2])

    assert str(hexa.uuid) == str(uns.uuid), "returned mesh uuid must match"
    np.testing.assert_allclose(points, hexa.points_ref())  # type: ignore

    np.testing.assert_allclose(nodes_per_face, hexa.nodes_per_face)
    np.testing.assert_allclose(nodes_per_face_cl, hexa.nodes_per_face_cl)
    np.testing.assert_allclose(faces_per_cell, hexa.faces_per_cell)
    np.testing.assert_allclose(faces_per_cell_cl, hexa.faces_per_cell_cl)
    np.testing.assert_allclose(cell_face_is_right_handed, hexa.cell_face_is_right_handed)

    for key, value in prop_uris.items():
        propkind_uri = value[0]
        prop_uri = value[1]
        prop0, values = await eclient.get_epc_mesh_property(rddms_uris[0], prop_uri[0])
        # print(f"property {key}: array size {values.shape}, mean {np.nanmean(values)}")
        assert prop0.supporting_representation.uuid == str(uns.uuid), "property support must match the mesh"

        prop_uuid = model.uuid(title=key)
        prop = rqp.Property(model, uuid=prop_uuid)
        continuous = prop.is_continuous()
        assert isinstance(prop0, ContinuousProperty) == continuous, "property types must match"
        assert isinstance(prop0, DiscreteProperty) == (not continuous), "property types must match"
        np.testing.assert_allclose(prop.array_ref(), values)   # type: ignore
