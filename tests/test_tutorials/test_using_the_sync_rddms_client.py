import examples.tutorials.using_the_sync_rddms_client.using_the_sync_rddms_client as mod


def test_using_the_sync_rddms_client() -> None:
    assert len(mod.dataspaces) == 1
    assert mod.dataspaces[0].path == "rddms_io/sync-demo"
    assert len(mod.gri_resources) == 1

    gri_uri = mod.gri.get_etp_data_object_uri(mod.dataspaces[0].path)
    crs_uri = mod.crs.get_etp_data_object_uri(mod.dataspaces[0].path)
    assert gri_uri == mod.gri_resources[0].uri
    assert gri_uri == mod.gri_lo.start_uri
    assert (
        len(mod.gri_lo.target_resources) == 1
        and crs_uri == mod.gri_lo.target_resources[0].uri
    )
