"""Tests of the interactive PLD picker on the event-16 demo data (direct S)."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

from bayes2mt import io, pld

matplotlib.use("Agg")
pytest.importorskip("ipywidgets")
from bayes2mt.interactive import PLDPicker

DEMO = Path(__file__).resolve().parents[1] / "examples" / "data" / "event016"
PICKS = DEMO / "picks.csv"
HYPOCENTER = dict(evla=34.6628, evlo=126.4013, evdp=20.58)
STATIONS = ["AJD", "BGD", "HWSB"]

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture
def picker(tmp_path):
    return PLDPicker(DEMO, "S", picks=PICKS, output_dir=tmp_path, **HYPOCENTER)


def test_station_list(picker):
    az = [picker.entries[s].azimuth for s in picker.order]
    assert len(picker.order) == 41 and np.all(np.diff(az) >= 0)
    assert picker.entries["AJD"].channel == "EL" and picker.entries["IBA"].channel == "HG"


def test_accept_writes_the_retrieved_astfs(picker, tmp_path):
    for sta in STATIONS:
        picker.select(sta)
        picker.accept()
    picks = [p for p in pld.read_picks(PICKS) if p.phase == "S" and p.station in STATIONS]
    archived = {a.code: a for a in pld.retrieve_from_picks(picks, DEMO)}
    written, _ = io.load_event_astfs(tmp_path, "S")
    assert sorted(a.code for a in written) == sorted(STATIONS)
    for a in written:
        b = archived[a.code]
        assert a.t.shape == b.t.shape
        np.testing.assert_allclose(a.t, b.t, rtol=0, atol=1e-12)
        np.testing.assert_allclose(a.f, b.f, rtol=0, atol=1e-12)
        assert abs(a.vr - b.vr) < 1e-12


def test_picks_csv_round_trip(picker, tmp_path):
    for sta in STATIONS:
        picker.accept(sta)
    original = {q.station: q for q in pld.read_picks(PICKS) if q.phase == "S"}
    written = pld.read_picks(tmp_path / "picks.csv")
    assert sorted(q.station for q in written) == sorted(STATIONS)
    for q in written:
        for key in pld.PICK_COLUMNS:
            assert getattr(q, key) == getattr(original[q.station], key), (q.station, key)


def test_duration_table_matches_shipped_table(picker):
    for sta in STATIONS:
        picker.accept(sta)
    tab = picker.duration_table().set_index("code")
    pub = io.read_tauc_table(DEMO / "tauc_table.csv")
    pub = pub[pub["phase"] == "S"].set_index("code").loc[tab.index]
    assert list(tab.reset_index().columns) == io.TABLE_COLUMNS
    for col in ["latitude", "longitude", "azimuth", "vr", "tauc", "t1", "toa"]:
        np.testing.assert_allclose(tab[col], pub[col], rtol=0, atol=1e-9, err_msg=col)


def test_reject_removes_output_and_resume(picker, tmp_path):
    picker.accept("AJD")
    picker.accept("BGD")
    picker.reject("BGD")
    assert picker.status["BGD"] == "rejected"
    assert not (tmp_path / "S" / "direct" / "BGD_pld_stf.pkl").exists()
    assert [q.station for q in pld.read_picks(tmp_path / "picks.csv")] == ["AJD"]
    again = PLDPicker(DEMO, "S", output_dir=tmp_path)
    assert again.status["AJD"] == "accepted" and again.status["BGD"] == "pending"
    assert again.picks["AJD"] == picker.picks["AJD"]


def test_widget_callbacks(picker):
    assert picker.ui is not None               # builds the widgets
    w = picker._w
    first, chosen = picker.current, picker.pick.duration
    w["duration"].value = 0.2
    assert picker.pick.duration == 0.2
    w["duration"].value = chosen
    w["accept"].click()
    assert picker.status[first] == "accepted" and picker.current != first
    w["station"].value = "AJD"
    assert picker.current == "AJD" and w["start"].value == -0.01 and w["duration"].value == 0.11
    w["reject"].click()
    assert picker.status["AJD"] == "rejected"


def test_settings_change_recomputes(picker):
    """Cached windows must not freeze start/duration: every setting change
    gives the ASTF that a fresh run of the same pick gives."""
    picker.select("AJD")
    for start, duration in [(-0.01, 0.05), (-0.01, 0.30), (-0.03, 0.11), (-0.01, 0.11)]:
        picker.set_params(start=start, duration=duration)
        got = picker.compute()
        main, egf = io.sac_pair_paths(DEMO, "AJD", "EL", "T", "KG")
        ref = pld.retrieve_astf(pld.prepare_windows(main, egf, picker.pick))
        np.testing.assert_array_equal(got.f, ref.f)
        assert got.vr == ref.vr


def test_set_params_validation(picker):
    with pytest.raises(TypeError):
        picker.set_params(durations=0.1)
    picker.set_params(start=-0.010000000000000002)
    assert picker.pick.start == -0.01


def test_sliders_redraw_and_save_table(picker, tmp_path):
    assert picker.ui is not None               # builds the widgets
    w = picker._w
    picker.select("AJD")
    before = w["figure"].value
    w["duration"].value = 0.20                 # a slider move recomputes and redraws
    assert picker.pick.duration == 0.20 and w["figure"].value != before
    w["start"].value = -0.03
    assert picker.pick.start == -0.03
    picker.set_params(start=-0.01, duration=0.11)
    w["accept"].click()
    w["save"].click()
    saved = pd.read_csv(tmp_path / "tauc_table_S.csv")
    assert list(saved.code) == ["AJD"] and "Saved the duration table" in w["message"].value
    np.testing.assert_allclose(saved.tauc, picker.duration_table().tauc)


def test_slider_widens_for_a_saved_value_outside_its_range(picker):
    assert picker.ui is not None               # builds the widgets
    w = picker._w
    picker.select("AJD")
    picker.set_params(duration=0.55)           # beyond the default maximum 0.40
    picker._refresh()
    assert w["duration"].max >= 0.55 and w["duration"].value == 0.55


def test_live_figure_is_updated_in_place(picker):
    """Slider steps reuse the station's figure (fixed layout) and change the
    image; another station builds a new figure."""
    assert picker.ui is not None
    w = picker._w
    picker.select("AJD")
    fig, before = picker._view["fig"], w["figure"].value
    w["duration"].value = 0.20
    assert picker._view["fig"] is fig and w["figure"].value != before
    assert fig.get_layout_engine() is None or type(fig.get_layout_engine()).__name__ == "PlaceHolderLayoutEngine"
    picker.select("BGD")
    assert picker._view["fig"] is not fig and picker._view_station == "BGD"
