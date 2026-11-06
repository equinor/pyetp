import examples.tutorials.using_the_sync_rddms_client.using_the_sync_rddms_client as mod


def test_using_the_sync_rddms_client() -> None:
    assert "rddms_io/sync-demo" in [d.path for d in mod.dataspaces]
    dataspace = next(filter(lambda d: d.path == "rddms_io/sync-demo", mod.dataspaces))
    assert len(mod.gri_resources) == 1

    gri_uri = mod.gri.get_etp_data_object_uri(dataspace.path)
    crs_uri = mod.crs.get_etp_data_object_uri(dataspace.path)
    assert gri_uri == mod.gri_resources[0].uri
    assert gri_uri == mod.gri_lo.start_uri
    assert (
        len(mod.gri_lo.target_resources) == 1
        and crs_uri == mod.gri_lo.target_resources[0].uri
    )
