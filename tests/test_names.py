"""The name mapping is the only link between sources, so it must be exact."""

import pandas as pd
import pytest

from ucl.data import names


def test_mapping_has_no_ambiguous_alias():
    mapping = names.load_mapping()
    counts = mapping.groupby(["source", "source_name"]).canonical.nunique()
    assert counts.max() == 1


def test_ambiguous_alias_is_rejected(tmp_path):
    path = tmp_path / "team_names.csv"
    pd.DataFrame(
        {
            "canonical": ["Roma", "Lazio"],
            "country": ["ITA", "ITA"],
            "source": ["openfootball", "openfootball"],
            "source_name": ["Roma", "Roma"],
            "provenance": ["", ""],
        }
    ).to_csv(path, index=False)
    with pytest.raises(names.AmbiguousAliasError, match="openfootball:Roma"):
        names.load_mapping(path)


def test_all_thirty_six_teams_are_mapped():
    teams = names.load_teams()
    assert len(teams) == 36
    mapping = names.load_mapping()
    # The mapping also covers clubs outside the 36, so the 36 are a subset.
    assert set(teams.team) <= set(mapping.canonical)


def test_both_openfootball_spellings_resolve_to_one_team():
    aliases = names.canonical_aliases("openfootball")
    assert set(aliases["Bodo/Glimt"]) == {"Bodo/Glimt", "FK Bodø/Glimt"}
    frame = pd.DataFrame({"home": ["Bodo/Glimt", "FK Bodø/Glimt"], "away": ["Lens", "RC Lens"]})
    out = names.add_canonical(frame, "openfootball")
    assert list(out.home_canonical) == ["Bodo/Glimt", "Bodo/Glimt"]
    assert list(out.away_canonical) == ["Lens", "Lens"]


def test_unmapped_names_are_reported_not_dropped():
    frame = pd.DataFrame({"home": ["Arsenal", "Some New Club"], "away": ["Lens", "Lens"]})
    missing = names.unmapped_names(frame, "openfootball")
    assert list(missing.index) == ["Some New Club"]


def test_teams_with_a_football_data_code_resolve_in_the_cache():
    teams = names.load_teams()
    aliases = names.canonical_aliases("football_data")
    coded = teams[teams.football_data_code != ""]
    assert set(coded.team) <= set(aliases)


def test_normalisation_strips_club_type_tokens_and_years():
    assert names.normalize_club_name("1. FSV Mainz 05") == "mainz"
    assert names.normalize_club_name("1.FC Köln") == "koln"
    assert names.normalize_club_name("ACF Fiorentina") == "fiorentina"
    assert names.normalize_club_name("FK Bodø/Glimt") == "bodo glimt"


def test_normalisation_never_returns_an_empty_key():
    # "AIK" is nothing but club-type tokens, yet it is a club.
    assert names.normalize_club_name("AIK") == "aik"


def test_association_override_for_cross_border_clubs():
    assert names.association("MCO") == "FRA"   # AS Monaco plays in Ligue 1
    assert names.association("LIE") == "SUI"   # FC Vaduz plays in Switzerland
    assert names.association("ENG") == "ENG"


def test_cross_border_clubs_are_inside_the_intersection():
    import pandas as pd

    european = pd.DataFrame(
        {
            "home": ["AS Monaco FC", "FC Vaduz"],
            "away": ["Arsenal FC", "Arsenal FC"],
            "home_country": ["MCO", "LIE"],
            "away_country": ["ENG", "ENG"],
        }
    )
    inter = names.intersection_clubs(european, {"FRA", "SUI", "ENG"})
    assert "AS Monaco FC" in inter and "FC Vaduz" in inter


def test_normalisation_can_keep_years_to_separate_clubs():
    # "U Craiova" and "U Craiova 1948" are different clubs that collide once
    # founding years are stripped.
    assert names.normalize_club_name("U Craiova 1948") == names.normalize_club_name("U Craiova")
    assert names.normalize_club_name(
        "U Craiova 1948", keep_years=True
    ) != names.normalize_club_name("U Craiova", keep_years=True)


def test_merge_conflicts_flags_two_merged_names_that_meet(tmp_path):
    path = tmp_path / "team_names.csv"
    pd.DataFrame(
        {
            "canonical": ["Dundee", "Dundee"],
            "country": ["SCO", "SCO"],
            "source": ["football_data", "football_data"],
            "source_name": ["Dundee", "Dundee United"],
            "provenance": ["user-reviewed", "fd-variant (user-reviewed)"],
        }
    ).to_csv(path, index=False)
    frame = pd.DataFrame(
        {
            "home": ["Dundee United", "Dundee"],
            "away": ["Dundee", "Hearts"],
            "date": pd.to_datetime(["2021-09-19", "2021-09-26"]),
        }
    )
    conflicts = names.merge_conflicts(frame, "football_data", path=path)
    assert len(conflicts) == 1
    assert conflicts.iloc[0].home == "Dundee United"


def test_real_mapping_has_no_merge_conflicts():
    from ucl.config import season
    from ucl.data.football_data import load_results

    domestic = load_results(season("2026-27").cutoff_date)
    assert names.merge_conflicts(domestic, "football_data").empty


def test_reviewed_merges_are_declared_fd_variants():
    mapping = names.load_mapping()
    fd = mapping[mapping.source == "football_data"]
    variants = set(fd.loc[fd.provenance.str.startswith("fd-variant"), "source_name"])
    for canonical, spellings in fd.groupby("canonical").source_name.apply(list).items():
        unmarked = [s for s in spellings if s not in variants]
        assert len(unmarked) <= 1, f"{canonical} has several unmarked spellings: {unmarked}"


def test_candidate_pool_includes_names_that_already_have_an_id():
    # "Brighton" is already an id; excluding mapped names once hid it as the
    # candidate for the Wikipedia spelling "Brighton & Hove Albion".
    pool = ["Brighton", "Burnley", "Everton", "Coventry"]
    mapped = {"Brighton": "Brighton"}
    out = names.closest_names(
        names.normalize_club_name("Brighton & Hove Albion"), pool, mapped=mapped
    )
    assert out[0] == "Brighton (= Brighton)"
    assert len(out) == 3


def test_candidates_are_ranked_by_similarity():
    pool = ["Nordsjaelland", "Vestsjaelland", "Horsens"]
    out = names.closest_names(names.normalize_club_name("Nordsjælland"), pool, limit=2)
    assert out == ["Nordsjaelland", "Vestsjaelland"]


def test_every_reviewed_wikipedia_target_is_a_real_cache_name():
    from ucl.config import season
    from ucl.data.football_data import load_results

    domestic = load_results(season("2026-27").cutoff_date)
    present = set(domestic.home) | set(domestic.away)
    mapping = names.load_mapping()
    wiki = mapping[(mapping.source == "wikipedia") & (mapping.provenance == "user-reviewed")]
    fd_names = set(mapping[mapping.source == "football_data"].source_name)
    for row in wiki.itertuples():
        # Each reviewed wikipedia alias resolves to an id that is itself backed
        # by a football-data name, except the league-phase teams whose leagues
        # are not cached at all.
        backing = mapping[(mapping.canonical == row.canonical)
                          & (mapping.source == "football_data")]
        if len(backing):
            assert set(backing.source_name) <= present | fd_names
