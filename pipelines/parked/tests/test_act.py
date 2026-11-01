"""Shader formulas match the host reference on one draw. No checkpoint required."""

from test_packing import pipeline


def test_activation_shaders_match_the_host_formulas() -> None:
    pipeline.check_act_kernels()
    pipeline.check_routed_expert()
    pipeline.check_gdn_scan()
