from test_packing import pipeline


def test_fp4_and_fp8_prefill_and_decode_match_the_dot() -> None:
    report = pipeline.check_int4_gemm()
    assert report["prefill"]
    assert report["fp8"]
    assert report["decode"]
    assert report["wide"]
    assert report["ok"]
