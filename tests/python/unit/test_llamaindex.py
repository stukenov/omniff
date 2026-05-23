from omniff.integrations.llamaindex import OmniFFReader


def test_reader_init():
    reader = OmniFFReader()
    assert reader.base_url == "http://localhost:8000"
    assert ".jpg" in reader.supported_extensions


def test_can_read_image():
    reader = OmniFFReader()
    assert reader.can_read("photo.jpg")
    assert reader.can_read("doc.pdf")
    assert not reader.can_read("script.py")


def test_custom_extensions():
    reader = OmniFFReader(supported_extensions=[".csv"])
    assert reader.can_read("data.csv")
    assert not reader.can_read("image.png")
