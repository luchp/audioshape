"""Tests for the VituixCAD export module."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from audioshape.database import parse_database
from audioshape.driver import Driver
from audioshape.ranking import evaluate
from audioshape.scenario import Scenario
from audioshape.vituixcad import RoleSelection, driver_database_tsv, project_xml


@pytest.fixture
def driver_s() -> Driver:
    """Sub-suited driver (paper's S): huge Vd, low Fs, low f_L."""
    return Driver(
        manufacturer="Example", model="S18", size_in=18,
        fs=20.0, qes=0.543, qms=4.0, re=3.5, mms=0.400,
        sd=0.115, xmax=0.020, vas=0.297, p_max=600.0, bl=18.0, le=4e-3,
        type_code="S")


@pytest.fixture
def driver_m() -> Driver:
    """Attack-suited driver (paper's M): small Vd, high Fs, high f_L. No
    Bl/Le -- exercises the "unknown" (NaN) path of the TSV round-trip."""
    return Driver(
        manufacturer="B&C Speakers", model="M/12\"", size_in=12,
        fs=48.0, qes=0.353, qms=6.0, re=5.2, mms=0.065,
        sd=0.052, xmax=0.008, vas=0.0648, p_max=350.0)


@pytest.fixture
def scenario() -> Scenario:
    return Scenario(v_room=60.0, l_max=6.0, r_listen=3.0,
                    sub_target_spl=110.0, attack_target_spl=105.0,
                    qtc=0.55, f_low=15.0, f_split=80.0, f_high=250.0)


@pytest.fixture
def selections(driver_s, driver_m, scenario) -> list[RoleSelection]:
    ev_sub = evaluate(driver_s, scenario, n_units=2,
                      band_low=scenario.f_low, band_high=scenario.f_split,
                      doppler_ref=scenario.f_split)
    ev_attack = evaluate(driver_m, scenario, n_units=1,
                         band_low=scenario.f_split, band_high=scenario.f_high,
                         doppler_ref=scenario.f_high, role="attack")
    return [
        RoleSelection("sub", ev_sub, scenario.f_low, scenario.f_split),
        RoleSelection("attack", ev_attack, scenario.f_split, scenario.f_high),
    ]


# ----------------------------------------------------------------------
# driver_database_tsv
# ----------------------------------------------------------------------

def test_tsv_round_trips_through_parser(driver_s, driver_m, tmp_path):
    text = driver_database_tsv([driver_s, driver_m])
    path = tmp_path / "VituixCAD_Drivers_selection.txt"
    path.write_text(text, encoding="utf-8")

    result = parse_database(path)
    assert not result.skipped
    by_label = {d.label(): d for d in result.drivers}
    assert set(by_label) == {"Example S18", 'B&C Speakers M/12"'}

    back = by_label["Example S18"]
    assert back.fs == pytest.approx(driver_s.fs)
    assert back.qes == pytest.approx(driver_s.qes)
    assert back.qms == pytest.approx(driver_s.qms)
    assert back.qts == pytest.approx(driver_s.qts)
    assert back.vas == pytest.approx(driver_s.vas)
    assert back.sd == pytest.approx(driver_s.sd)
    assert back.xmax == pytest.approx(driver_s.xmax)
    assert back.bl == pytest.approx(driver_s.bl)
    assert back.le == pytest.approx(driver_s.le)
    assert back.type_code == "S"

    # driver_m has no Bl/Le (NaN) -- must round-trip as blank, not "nan"
    back_m = by_label['B&C Speakers M/12"']
    assert not (back_m.bl == back_m.bl)  # NaN
    assert not (back_m.le == back_m.le)


def test_tsv_deduplicates_by_label(driver_s):
    text = driver_database_tsv([driver_s, driver_s])
    assert text.count("Example\tS18") == 1


# ----------------------------------------------------------------------
# project_xml
# ----------------------------------------------------------------------

def test_project_xml_well_formed_and_has_both_drivers(selections):
    text = project_xml(selections, description="hello & <world>")
    assert text.startswith('<?xml version="1.0" encoding="utf-8"?>')
    assert "<!--VituixCAD PROJECT-->" in text

    # strip the leading XML decl + comments (ElementTree can't parse
    # top-level comments) before parsing the element tree itself
    body = text.split("-->", 2)[-1]
    root = ET.fromstring(body)

    assert root.tag == "SPEAKER"
    assert root.findtext("Description") == "hello & <world>"
    assert root.findtext("HalfSpace") == "True"
    assert root.find("AxialTarget") is None
    assert root.find("PowerTarget") is None

    models = [e.findtext("Model") for e in root.findall("DRIVER")]
    assert models == ["Example S18", 'B&C Speakers M/12"']

    driver_parts = [p for p in root.find("CROSSOVER").findall("PART")
                    if p.findtext("Type") == "Driver"]
    assert len(driver_parts) == 2
    drv_n = [p.find("DriverTarget").findtext("DrvN") for p in driver_parts]
    assert drv_n == ["2", "1"]  # sub_units=2, attack_units=1

    # role bands land on the driver's own target, not the whole scenario
    sub_part = driver_parts[0]
    assert sub_part.find("DriverTarget").findtext("FreqMin") == "15"
    assert sub_part.find("DriverTarget").findtext("FreqMax") == "80"
    assert sub_part.find("DriverTarget").findtext("SPL") == "110"
    attack_part = driver_parts[1]
    assert attack_part.find("DriverTarget").findtext("FreqMin") == "80"
    assert attack_part.find("DriverTarget").findtext("FreqMax") == "250"
    assert attack_part.find("DriverTarget").findtext("SPL") == "105"


def test_project_xml_requires_at_least_one_selection():
    with pytest.raises(ValueError):
        project_xml([])


def test_project_xml_single_role(driver_s, scenario):
    ev = evaluate(driver_s, scenario, n_units=1)
    text = project_xml([RoleSelection("full", ev, scenario.f_low, scenario.f_high)])
    body = text.split("-->", 2)[-1]
    root = ET.fromstring(body)
    assert len(root.findall("DRIVER")) == 1
    driver_parts = [p for p in root.find("CROSSOVER").findall("PART")
                    if p.findtext("Type") == "Driver"]
    assert len(driver_parts) == 1
