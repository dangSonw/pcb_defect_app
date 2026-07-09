from ui.tab_run import apply_plc_result_state


class DummyPLC:
    def __init__(self):
        self.calls = []

    def batchwrite_bitunits(self, address, values):
        self.calls.append((address, values))


def test_apply_plc_result_state_sets_ok_bits():
    plc = DummyPLC()

    apply_plc_result_state(plc, has_defect=False)

    assert plc.calls == [
        ("M169", [0, 1, 0]),
        ("M170", [0]),
    ]


def test_apply_plc_result_state_sets_ng_bits():
    plc = DummyPLC()

    apply_plc_result_state(plc, has_defect=True)

    assert plc.calls == [
        ("M169", [0, 0, 1]),
        ("M171", [0]),
    ]
