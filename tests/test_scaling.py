import math


def test_uniform_is_log2_27():
    from src.train import uniform_baseline
    assert abs(uniform_baseline() - math.log2(27)) < 1e-9
    assert 4.7 < uniform_baseline() < 4.8


def test_floor_fit_recovers_video_numbers():
    import numpy as np
    from scripts.run_study import PUBLISHED_N, PUBLISHED_L
    from src.scaling import fit_floor_power, predict
    fit = fit_floor_power(PUBLISHED_N[:5], PUBLISHED_L[:5, 0])
    # video: floor ~1.89, per4x ~0.74-0.77
    assert 1.5 < fit["floor"] < 2.1, fit
    assert 0.6 < fit["per4x"] < 0.85, fit
    preds = predict(fit, PUBLISHED_N[5:])
    err = float(np.abs(preds - PUBLISHED_L[5:, 0]).max())
    assert err < 0.15, err  # video claims <=0.02; allow slack for reconstructed table


def test_joint_fit_shape():
    import numpy as np
    from scripts.run_study import PUBLISHED_N, PUBLISHED_D, PUBLISHED_L
    from src.scaling import fit_joint
    Nn, Dd, Ll = [], [], []
    for i, n in enumerate(PUBLISHED_N):
        for j, d in enumerate(PUBLISHED_D):
            Nn.append(n); Dd.append(d); Ll.append(PUBLISHED_L[i, j])
    j = fit_joint(np.array(Nn), np.array(Dd), np.array(Ll))
    assert 1.2 < j["E"] < 1.8
    assert j["mae"] < 0.10
    assert j["maxe"] < 0.20
