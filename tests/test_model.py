def test_model_counts_and_generate():
    import torch
    from src.model import LetterGPT, count_params
    m = LetterGPT(vocab=27, context=16, d_model=16, n_layer=1, n_head=2)
    assert count_params(m) > 500
    assert count_params(m, exclude_embedding=True) < count_params(m)
    m.eval()
    idx = torch.zeros((1, 4), dtype=torch.long)
    out = m.generate(idx, 10)
    assert out.shape == (1, 14)


def test_normalize_27_letters():
    from src.data import normalize, VOCAB
    s = normalize("Hello, World 48!")
    assert set(s) <= set(VOCAB + [" "]) or set(s) <= set(VOCAB)
    assert "foureight" in s  # 48 spelled digit-by-digit
