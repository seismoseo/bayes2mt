"""Tests of the plotting helpers that do not depend on the demo data."""
import matplotlib
import numpy as np

matplotlib.use("Agg")
from bayes2mt.plotting import corner

RNG = np.random.default_rng(620)
N = 20000


def _posterior(**over):
    post = dict(tauc=RNG.normal(0.045, 0.0015, N), Lc=RNG.normal(208, 10, N),
                Wc=np.abs(RNG.normal(70, 30, N)) + 1, v0=RNG.normal(2980, 250, N),
                rake=RNG.normal(250, 3, N) % 360, angle=RNG.normal(73, 4, N) % 180,
                dir_ratio=RNG.normal(0.65, 0.05, N), stress_drop=10 ** RNG.normal(1.8, 0.5, N))
    post.update(over)
    return post


def test_auto_panel_ticks_fit_and_stay_inside():
    width_pt = 72 * 24 * (0.985 - 0.085) / (8 + 0.16 * 7)
    for scale in (0.1, 1, 30, 300):
        post = _posterior(tauc=0.045 * scale + RNG.normal(0, 0.0015 * scale, N),
                          Lc=208 * scale + RNG.normal(0, 10 * scale, N))
        for p in corner.auto_panels(post):
            lo, hi = p.limits
            assert lo < hi and 2 <= len(p.major) <= 5
            assert all(lo <= v <= hi for v in p.major), (p.key, p.limits, p.major)
            assert len(set(p.ticklabels)) == len(p.ticklabels), (p.key, p.ticklabels)
            if not p.log:                      # neighbouring labels do not overlap
                step_pt = (p.major[1] - p.major[0]) / (hi - lo) * width_pt
                assert step_pt >= max(corner._label_width(t, 22) for t in p.ticklabels), p.key


def test_auto_panels_unwrap_angles_into_zero_to_period():
    post = _posterior(rake=RNG.normal(2, 8, N) % 360)
    panels = {p.key: p for p in corner.auto_panels(post)}
    assert 0 <= panels["rake"].center < 360 and 0 <= panels["angle"].center < 180
    lo, hi = panels["rake"].limits
    assert lo < 0 < hi                          # across 0/360: one lobe, not two
    x = corner._values(panels["rake"], post)
    assert np.mean((x > lo) & (x < hi)) > 0.99


def test_retrieval_figure_keeps_the_interactive_flag(monkeypatch):
    """IPython switches interactive mode on when the session's first figure is
    created.  If plot_astf_retrieval restored every rcParam afterwards (as
    rc_context does), interactive mode would be off again and all later inline
    figures of the notebook would not be shown."""
    from bayes2mt.plotting import retrieval

    class Stop(Exception):
        pass

    def first_figure(*args, **kwargs):          # what IPython does on the first figure
        matplotlib.interactive(True)
        raise Stop

    monkeypatch.setattr(retrieval.plt, "figure", first_figure)
    font_size = matplotlib.rcParams["font.size"]
    matplotlib.interactive(False)
    try:
        try:
            retrieval.plot_astf_retrieval(*[None] * 10)
        except Stop:
            pass
        assert matplotlib.is_interactive()
        assert matplotlib.rcParams["font.size"] == font_size
    finally:
        matplotlib.interactive(False)
