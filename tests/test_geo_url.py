from scripts.download_geo import geo_range, archive_url


def test_geo_range():
    assert geo_range("GSE292268") == "GSE292nnn"
    assert geo_range("GSE1000") == "GSE1nnn"


def test_archive_url():
    assert archive_url("GSE292268") == "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE292nnn/GSE292268/suppl/GSE292268_RAW.tar"
