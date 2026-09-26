from src.normalization import normalize_text


TESTS = [
    ("MS [Consultancy]", "ms consultancy"),
    ("MS Consultancy Corp", "ms consultancy corp"),
    ("Clairvoyant Récord Private Ltd", "clairvoyant récord private ltd"),
    ("Clairvoyant Record Private Limited", "clairvoyant record private limited"),
    ("Davis Family Offie", "davis family offie"),
    ("Davis Family Office", "davis family office"),
    ("Offiec No S ##07 82Haware", "offiec no s 07 82haware"),
    ("Office No S 07 82Haware", "office no s 07 82haware"),
    ("सुप्रीम आईटी प्राइवेट लिमिटेड", "सुप्रीम आईटी प्राइवेट लिमिटेड"),
]


def test_phase7_normalization():
    for raw, expected in TESTS:
        assert normalize_text(raw) == expected
