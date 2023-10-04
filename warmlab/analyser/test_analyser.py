from analyser import analyser
from warmlab import warm


def test_escape_string():
    pass


def test_parse_model():
    analyser.parse_model(warm.WarmModel(
        "ring_2d_3",
        is_graph=True,
        graph=warm.ring_2d_graph(3)
    ))


def test_fetch_data():
    data = analyser.fetch_data("ring_2d")
    print(data)
