import pytest

from datacard_viewer.cli import card_parser


def test_cache_limit_defaults():
    args = card_parser().parse_args([])
    assert args.max_files == 8
    assert args.max_mb == 128


@pytest.mark.parametrize(
    "option,value",
    [("--max-files", "1.5"), ("--max-mb", "1.5")],
)
def test_invalid_cache_limits(option, value):
    with pytest.raises(SystemExit) as error:
        card_parser().parse_args([option, value])
    assert error.value.code == 2
