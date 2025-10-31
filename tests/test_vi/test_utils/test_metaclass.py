from torch_blue.vi.utils import PostInitCallMeta


def test_post_init_meta() -> None:
    """Test PostInitCallMeta."""

    class Test(metaclass=PostInitCallMeta):
        a: int

        def __post_init__(self) -> None:
            if not hasattr(self, "a"):
                raise NotImplementedError("Test error")

    try:
        _ = Test()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Test error"

    class Test2(Test):
        def __init__(self, a: int) -> None:
            self.a = a

    Test2(3)
