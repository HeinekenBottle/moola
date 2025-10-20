def test_import():
    import moola

    assert hasattr(moola, "__version__")
