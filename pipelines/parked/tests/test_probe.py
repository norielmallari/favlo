from test_packing import pipeline


def test_block_roundtrip_matches_this_gpu() -> None:
    report = pipeline.Roundtrip().measure()
    assert report["ok"], report

    int4 = report["formats"]["int4"]
    assert int4["element_order"] == "low_nibble_is_element_0"
    assert int4["signed_codes"]["0"] == 0
    assert int4["signed_codes"]["7"] == 7
    assert int4["signed_codes"]["8"] == -8
    assert int4["signed_codes"]["15"] == -1
    assert int4["packed_bytes"] == 16

    int2 = report["formats"]["int2"]
    assert int2["ok"]
    assert int2["packed_bytes"] == 8
    assert int2["element_order"] is None

    assert report["formats"]["int8"]["logical_matches_pattern"]
    assert report["formats"]["fp8_e4m3"]["shader_matches_ocp"]
    assert report["formats"]["fp8_e5m2"]["shader_matches_ocp"]

    scale = report["scale"]
    assert scale["scale_read"] == 130
    assert scale["storage"] == "low_nibble_is_element_0"
    assert scale["shader_matches_reconstruct"]
