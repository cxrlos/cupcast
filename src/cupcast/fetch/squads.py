from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

from cupcast.fetch.tls import ensure_system_certificates

# Raw wikitext of the consolidated squads page: every player is a structured
# {{nat fs g player|...}} template, far more reliable than scraping HTML.
SQUADS_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads?action=raw"
TOURNAMENT_START = pd.Timestamp("2026-06-11")

GROUP_RE = re.compile(r"^==\s*Group ([A-L])\s*==\s*$", re.MULTILINE)
TEAM_RE = re.compile(r"^===\s*\{?\{?[^=]*?([A-Za-zÀ-ž' .&-]+?)\}?\}?\s*===\s*$", re.MULTILINE)
PLAYER_RE = re.compile(r"\{\{nat fs (?:g )?player\s*\|(?P<body>.*)$", re.IGNORECASE)
COACH_RE = re.compile(r"Head coach:.*?\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")

NO_RE = re.compile(r"\bno=(\d+)")
POS_RE = re.compile(r"\bpos=([A-Z]+)")
NAME_RE = re.compile(r"\bname=\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
NAME_PLAIN_RE = re.compile(r"\bname=([^|}\[]+)")
BIRTH_RE = re.compile(
    r"\|\s*\d{4}\s*\|\s*\d{1,2}\s*\|\s*\d{1,2}\s*\|\s*(\d{4})\s*\|\s*(\d{1,2})\s*\|\s*(\d{1,2})"
)
CAPS_RE = re.compile(r"\bcaps=(\d+)")
GOALS_RE = re.compile(r"\bgoals=(\d+)")
CLUB_RE = re.compile(r"\bclub=\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
CLUBNAT_RE = re.compile(r"\bclubnat=([A-Za-z]{3})")


def fetch_squads_wikitext(
    cache_dir: str | Path = "data/raw/wikipedia",
    session: requests.Session | None = None,
    refresh: bool = False,
) -> str:
    cache_path = Path(cache_dir) / "2026_squads.wikitext"
    if cache_path.exists() and not refresh:
        return cache_path.read_text()
    ensure_system_certificates()
    response = (session or requests).get(
        SQUADS_URL, timeout=60, headers={"User-Agent": "cupcast/0.1 (research project)"}
    )
    response.raise_for_status()
    text = response.text
    if "nat fs" not in text:
        raise OSError("squads wikitext missing player templates; page layout changed?")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text)
    return text


def _parse_player(body: str) -> dict | None:
    name_match = NAME_RE.search(body) or NAME_PLAIN_RE.search(body)
    if name_match is None:
        return None
    birth = BIRTH_RE.search(body)
    birth_date = (
        pd.Timestamp(int(birth.group(1)), int(birth.group(2)), int(birth.group(3)))
        if birth
        else pd.NaT
    )
    club = CLUB_RE.search(body)
    return {
        "number": int(NO_RE.search(body).group(1)) if NO_RE.search(body) else None,
        "position": POS_RE.search(body).group(1) if POS_RE.search(body) else None,
        "name": name_match.group(1).strip(),
        "birth_date": birth_date,
        "caps": int(CAPS_RE.search(body).group(1)) if CAPS_RE.search(body) else 0,
        "goals": int(GOALS_RE.search(body).group(1)) if GOALS_RE.search(body) else 0,
        "club": club.group(1).strip() if club else None,
        "club_country": (
            CLUBNAT_RE.search(body).group(1).upper() if CLUBNAT_RE.search(body) else None
        ),
    }


def parse_squads(wikitext: str) -> pd.DataFrame:
    rows = []
    group = None
    team = None
    coach = None
    for line in wikitext.splitlines():
        group_match = GROUP_RE.match(line)
        if group_match:
            group = group_match.group(1)
            continue
        team_match = TEAM_RE.match(line)
        if team_match and group is not None:
            team = team_match.group(1).strip()
            coach = None
            continue
        coach_match = COACH_RE.search(line)
        if coach_match:
            coach = coach_match.group(1).strip()
            continue
        player_match = PLAYER_RE.search(line)
        if player_match and team is not None:
            player = _parse_player(player_match.group("body"))
            if player is not None:
                age = (
                    (TOURNAMENT_START - player["birth_date"]).days / 365.25
                    if pd.notna(player["birth_date"])
                    else None
                )
                rows.append(
                    {"group": group, "team": team, "coach": coach, **player, "age": age}
                )
    return pd.DataFrame(rows)


def load_squads(
    cache_dir: str | Path = "data/raw/wikipedia", refresh: bool = False
) -> pd.DataFrame:
    return parse_squads(fetch_squads_wikitext(cache_dir, refresh=refresh))


def main(refresh: bool = False) -> None:
    squads = load_squads(refresh=refresh)
    out = Path("data/processed/squads.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    squads.to_csv(out, index=False)
    counts = squads.groupby("team").size()
    print(f"{len(squads)} players, {squads['team'].nunique()} teams -> {out}")
    short = counts[counts != 26]
    if not short.empty:
        print("teams without exactly 26 parsed players:")
        print(short.to_string())


if __name__ == "__main__":
    main()
