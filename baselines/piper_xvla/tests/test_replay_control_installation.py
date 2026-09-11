from piper_xvla.replay_control import INSTALLATION_CHOICES


def test_installation_choices_map_keep_to_zero():
    assert INSTALLATION_CHOICES == {
        "keep": 0x00,
        "horizontal": 0x01,
        "side-left": 0x02,
        "side-right": 0x03,
    }
