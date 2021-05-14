from eventio import EventIOFile
import pytest

testfile = 'tests/resources/one_shower.dat'


def test_iterator():
    f = EventIOFile(testfile)

    # make sure f is an iterator
    assert iter(f) is iter(f)

    # make sure, it does not restart:

    it = iter(f)
    first = next(it)

    for o in it:
        assert o.header.content_address > first.header.content_address


def test_peek():
    f = EventIOFile(testfile)

    o = f.peek()
    assert o is f.peek()  # make sure peek does not advance
    assert o is next(f)   # make sure peek gives us the next object
    assert o is not f.peek()  # assure we get the next
    assert f.peek() is next(f)

    # make sure peek returns None at end of file
    for o in f:
        pass

    assert f.peek() is None

    # make sure peek returns None at end of file also for truncated files
    truncated = 'tests/resources/gamma_test_large_truncated.simtel.gz'

    # make sure file was really truncated and we reached end of file
    with pytest.raises(EOFError):
        f = EventIOFile(truncated)
        for o in f:
            pass

    f = EventIOFile(truncated)

    while f.peek() is not None:
        o = next(f)

    # test we can peak multiple times for a truncated file
    assert f.peek() is None
    assert f.peek() is None
