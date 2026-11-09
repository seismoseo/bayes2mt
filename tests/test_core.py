"""Unit tests (no data) and tests on the event-16 demo data in examples/."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bayes2mt import geometry, io, moments, pld, relloc, stressdrop, synthetic

DEMO = Path(__file__).resolve().parents[1] / "examples" / "data" / "event016"

RNG = np.random.default_rng(0)


def _table(n=60, seed=1):
    rng = np.random.default_rng(seed)
    phase = rng.choice(["P", "S", "SmS"], n)
    toa = np.where(phase == "SmS", rng.uniform(20, 60, n), rng.uniform(95, 160, n))
    return pd.DataFrame(dict(azimuth=rng.uniform(0, 360, n), toa=toa, phase=phase))


def _m2_from(tauc, Lc, Wc, angle, vx, vy):
    """Moments of a model with given physical parameters (m, s, km/s, deg)."""
    th = np.deg2rad(angle)
    u = np.array([np.cos(th), -np.sin(th)])          # angle measured to -y (up dip)
    v = np.array([np.sin(th), np.cos(th)])
    S = (Lc / 2000) ** 2 * np.outer(u, u) + (Wc / 2000) ** 2 * np.outer(v, v)
    mu02 = (tauc / 2) ** 2
    return np.array([mu02, vx * mu02, vy * mu02, S[0, 0], S[0, 1], S[1, 1]])


# ------------------------------------------------------------ pure tests
def test_cholesky_parameterisation_is_psd_and_batched():
    x = RNG.uniform([-2, -1, -1, -2, -1, -2], [-0.4, 1, 1, 0, 1, 0], size=(500, 6))
    m2 = moments.cholesky_to_m2(x)
    assert np.allclose(m2[7], moments.cholesky_to_m2(x[7]))
    X = np.array([[[m[0], m[1], m[2]], [m[1], m[3], m[4]], [m[2], m[4], m[5]]] for m in m2])
    assert np.linalg.eigvalsh(X).min() > -1e-12


def test_physical_recovers_known_model():
    m2 = _m2_from(0.05, 300.0, 100.0, 30.0, 1.0, -2.0)
    p = moments.physical(m2)
    assert p["tauc"] == pytest.approx(0.05)
    assert p["Lc"] == pytest.approx(300.0)
    assert p["Wc"] == pytest.approx(100.0)
    assert p["angle"] == pytest.approx(30.0)
    assert p["rake"] == pytest.approx(np.rad2deg(np.arctan2(2.0, 1.0)))
    assert p["v0"] == pytest.approx(np.hypot(1, 2) * 1000)


def test_slowness_matrix_pure_duration_model_gives_constant_tauc():
    G = geometry.slowness_matrix(_table(), 108, 58, 6.2, 3.6)
    pred = moments.predict_tauc(G, np.array([0.02 ** 2, 0, 0, 0, 0, 0]))
    assert np.allclose(pred, 0.04)


def test_hdi_and_circular_hdi():
    x = np.linspace(0, 1, 100001)
    lo, hi = moments.hdi(x, 0.9)
    assert hi - lo == pytest.approx(0.9, abs=1e-4)
    ang = np.mod(RNG.normal(0, 5, 20000), 360)
    h = moments.circular_hdi(ang, 0.95, 360)
    assert h.shape == (2, 2) and h[0, 1] == 360 and h[1, 0] == 0


def test_medoid_matches_bruteforce():
    X = RNG.normal(size=(800, 6))
    X = np.vstack([X, X[:200]])                      # duplicates, as in MCMC output
    D = np.sqrt(((((X - X.mean(0)) / X.std(0))[:, None] - ((X - X.mean(0)) / X.std(0))[None]) ** 2).sum(-1))
    assert np.allclose(X[moments.medoid_index(X)], X[np.argmin(D.sum(1))])


def test_pld_recovers_boxcar():
    rng = np.random.default_rng(3)
    dt, n = 0.01, 151
    x = np.convolve(rng.normal(size=n), np.hanning(9), "same")
    true = np.zeros(n)
    true[75:81] = 1.0                                 # 60 ms boxcar at t = 0
    y = np.convolve(x, true, "same")
    f, t, vr = pld.pl_deconv(y, x, dt, -0.02, 0.10, len(x), iter_max=300)
    assert vr > 95
    assert pld.characteristic_duration(t, f) == pytest.approx(pld.characteristic_duration(t, true[:len(t)]), rel=0.15)


def test_tradeoff_curve_equals_separate_runs():
    rng = np.random.default_rng(4)
    dt, n = 0.01, 151
    x = np.convolve(rng.normal(size=n), np.hanning(9), "same")
    y = np.convolve(x, np.r_[np.zeros(75), np.ones(6), np.zeros(70)], "same") + 0.05 * rng.normal(size=n)
    durations = pld.default_durations(-0.03, dt)
    _, misfit = pld.tradeoff_curve(y, x, dt, -0.03, durations, iter_max=50)
    single = [1 - pld.pl_deconv(y, x, dt, -0.03, -0.03 + d, n, iter_max=50)[2] / 100 for d in durations]
    assert np.array_equal(misfit, single)


def test_stress_drop_closed_form():
    M0, L = 1e14, 200.0
    assert float(stressdrop.stress_drop(M0, L, L, 30, 0)) == pytest.approx(7 * M0 / (16 * L ** 3) / 1e6, rel=1e-12)
    assert float(stressdrop.circular_stress_drop(M0, L)) == pytest.approx(7 * M0 / (16 * L ** 3) / 1e6, rel=1e-12)
    along, across = (float(stressdrop.stress_drop(M0, 2 * L, L, r, 0)) for r in (0, 90))
    assert across > along                                   # slip across the long axis: larger drop
    assert float(stressdrop.stress_drop(M0, 2 * L, L, 45, 0)) == pytest.approx((along + across) / 2, rel=1e-12)
    assert float(stressdrop.stress_drop(M0, 2 * L, L, 200, 20)) == pytest.approx(along, rel=1e-12)
    assert float(stressdrop.stress_drop(M0, 2 * L, L, 0, 0, semi_axis_factor=np.sqrt(5) / 2)) == pytest.approx(
        along / (np.sqrt(5) / 2) ** 3, rel=1e-12)
    sd = stressdrop.stress_drop(M0, np.array([2 * L, 3 * L]), L, 10.0, np.array([5.0, 40.0]))
    assert sd.shape == (2,) and np.all(sd > 0)


def test_synthetic_crack_second_moments():
    pytest.importorskip("cvxpy")
    from bayes2mt import deterministic
    # elliptical slip D ~ sqrt(1 - r^2): second-moment length = (2 / sqrt 5) x semi-axis
    rup = synthetic.Rupture(strike=30, dip=90, semi_strike=0.2, semi_dip=0.1, rise_time=0.01,
                            n_cells=101, sampling_rate=1000.0, duration=0.3)
    t, f = synthetic.astfs(rup, [0.0, 90.0], [90.0, 90.0], 3.6)
    assert np.allclose(np.trapezoid(f, t, axis=1) if hasattr(np, "trapezoid") else np.trapz(f, t, axis=1), 1.0)
    tab = pd.DataFrame(dict(azimuth=np.tile(np.arange(0, 360, 20.0), 3), toa=np.repeat([60.0, 100.0, 140.0], 18),
                            phase="S"))
    tab = synthetic.tauc_table(rup, tab, 6.2, 3.6)
    r = deterministic.invert(tab, 30, 90, 6.2, 3.6, solver="CLARABEL")
    assert r.params["Lc"] == pytest.approx(2 / np.sqrt(5) * 200, rel=0.03)
    assert r.params["Wc"] == pytest.approx(2 / np.sqrt(5) * 100, rel=0.06)
    assert r.params["dir_ratio"] < 0.05                     # symmetric (bilateral) rupture
    noisy = synthetic.add_noise(tab, 0.1, seed=1)
    assert not np.allclose(noisy["tauc"], tab["tauc"])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_demo_duration_table_matches_shipped_table():
    picks = pld.read_picks(DEMO / "picks.csv")
    astfs = pld.retrieve_from_picks(picks, DEMO)
    tab = pld.duration_table(astfs, picks, DEMO, 34.6628, 126.4013, 20.58)
    ref = io.read_tauc_table(DEMO / "tauc_table.csv")
    for phase in ("P", "S", "SmS"):
        a = tab[tab.phase == phase].set_index("code")
        b = ref[ref.phase == phase].set_index("code").loc[a.index]
        assert len(a) == len(b) == (ref.phase == phase).sum() > 10
        for col in ("tauc", "vr", "t1", "azimuth", "toa"):
            # round-off differs between numpy/scipy builds (~1e-8 relative), so not bit-exact
            np.testing.assert_allclose(a[col], b[col], rtol=1e-6, atol=1e-9, err_msg=f"{phase} {col}")


def test_load_event_astfs_skips_empty_s_folder(tmp_path):
    """S ASTFs directly in S/ (next to an empty S/dirct) are found."""
    import pickle
    (tmp_path / "S" / "dirct").mkdir(parents=True)
    t = np.arange(50) * 0.01
    for code in ("AAA", "BBB"):
        with open(tmp_path / "S" / f"{code}_pld_stf.pkl", "wb") as f:
            pickle.dump([t, np.exp(-((t - 0.2) / 0.03) ** 2), 90.0], f)
    for name, obj in (("station_list.pkl", ["BBB", "AAA"]), ("azimuth_list.pkl", [200.0, 10.0])):
        with open(tmp_path / "S" / name, "wb") as f:
            pickle.dump(obj, f)
    astfs, az = io.load_event_astfs(tmp_path, "S")
    assert [a.code for a in astfs] == ["AAA", "BBB"] and list(az) == [10.0, 200.0]


def test_relloc_recovers_grid_point():
    tab = _table(80, seed=5)
    v = np.where(tab["phase"] == "P", 3600 * 1.73, 3600.0)
    tab["t1"] = relloc.predict_onsets(105.0, 110.0, -0.02, 70.0, tab["azimuth"], tab["toa"], v)
    r = relloc.locate(tab)
    assert (r.azimuth, r.dip, r.L) == pytest.approx((105.0, 110.0, 70.0), abs=1e-6)
    assert r.misfit < 1e-9


def test_sdp_recovers_noise_free_model():
    pytest.importorskip("cvxpy")
    from bayes2mt import deterministic
    tab = _table(80, seed=7)
    m2 = _m2_from(0.05, 250.0, 90.0, 70.0, -1.0, 2.5)
    tab["tauc"] = moments.predict_tauc(geometry.slowness_matrix(tab, 108, 58, 6.228, 3.6), m2)
    r = deterministic.invert(tab, 108, 58, 6.228, 3.6, solver="CLARABEL")
    assert r.vr > 99.99
    assert r.params["Lc"] == pytest.approx(250.0, rel=1e-3)


def test_bayesian_starting_states_are_seeded():
    pytest.importorskip("bayesbay")
    from bayes2mt import bayesian
    a = bayesian.starting_states(bayesian.Priors(), 3)
    b = bayesian.starting_states(bayesian.Priors(), 3)
    va = [a[i]["my_param_space"]["m0"][0] for i in range(3)]
    vb = [b[i]["my_param_space"]["m0"][0] for i in range(3)]
    rs = np.random.RandomState(620)
    assert va == vb and va[0] == rs.uniform(np.log10(0.01), np.log10(0.4))


def test_seeded_runs_repeat_and_continuations_do_not():
    pytest.importorskip("bayesbay")
    from bayes2mt import bayesian
    G = geometry.slowness_matrix(pd.DataFrame({"azimuth": np.arange(0, 360, 30.0), "toa": 60.0, "phase": "S"}),
                                 0.0, 90.0, 6.0, 3.5)
    tauc = moments.predict_tauc(G, moments.cholesky_to_m2([-1.3, 0.05, 0.0, -1.0, 0.0, -1.5]))

    def run(n_runs):
        inv = bayesian.build(G, tauc, n_chains=2)
        return [bayesian.run(inv, n_iterations=300, burnin_iterations=0, save_every=10, n_jobs=1)[1]
                for _ in range(n_runs)]

    first, = run(1)
    again, cont = run(2)
    assert np.array_equal(first, again)
    # a continued chain draws from a new stream, not the first call's again
    assert bayesian._chain_seed(620, 1, 0) == 621 != bayesian._chain_seed(620, 1, 300)
    assert cont.shape[1] > first.shape[1]
