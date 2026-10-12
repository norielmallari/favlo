from test_packing import pipeline


def test_int4_e8m0_prefill_and_decode_match_the_signed_dot() -> None:
    report = pipeline.check_int4_gemm()
    assert report["prefill"]
    assert report["decode"]
    assert report["wide"]
    assert report["ok"]
