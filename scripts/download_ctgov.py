"""Download a snapshot of ClinicalTrials.gov via the v2 REST API.

Uses the /studies endpoint with nextPageToken pagination. Filters at the
server side to interventional studies with non-empty eligibility criteria
to keep the snapshot manageable (~150-300k trials vs ~500k total).

Usage:
    python scripts/download_ctgov.py \\
        --output data/ctgov_snapshot \\
        --query "AREA[StudyType]Interventional" \\
        --page-size 1000 \\
        --max-pages 0      # 0 = no limit

Resumable: state is checkpointed every page to a state.json file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

API = "https://clinicaltrials.gov/api/v2/studies"

# httpx triggers ClinicalTrials.gov's TLS-fingerprint WAF (returns 403 even with
# a browser-like UA). `requests` and stdlib `urllib` work — we use requests for
# its session/retry ergonomics.


def fetch_page(session: requests.Session, params: dict, max_attempts: int = 5) -> dict:
    """GET /studies with exponential backoff on 5xx and 429."""
    delay = 2.0
    for attempt in range(max_attempts):
        try:
            r = session.get(API, params=params, timeout=60.0)
        except requests.RequestException as e:
            print(f"  network error attempt {attempt + 1}/{max_attempts}: {e}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            print(f"  retry {attempt + 1}/{max_attempts} (status={r.status_code})", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
    raise RuntimeError(f"Failed after {max_attempts} attempts on {params}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("data/ctgov_snapshot"))
    p.add_argument(
        "--query",
        type=str,
        default="Interventional",
        help="ClinicalTrials.gov v2 query.term value (Essie syntax)",
    )
    p.add_argument("--page-size", type=int, default=1000)
    p.add_argument("--max-pages", type=int, default=0, help="0 = no limit")
    p.add_argument("--resume", action="store_true", help="Resume from saved state.json")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "state.json"

    state = {"page": 0, "next_page_token": None, "total_studies": 0}
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text())
        print(f"Resuming from page {state['page']} (token={state['next_page_token']})")

    base_params: dict = {
        "query.term": args.query,
        "pageSize": args.page_size,
        "format": "json",
        # Field selection — drop heavy fields we won't use
        "fields": ",".join(
            [
                "NCTId",
                "BriefTitle",
                "OfficialTitle",
                "BriefSummary",
                "DetailedDescription",
                "Condition",
                "Keyword",
                "Phase",
                "OverallStatus",
                "StudyType",
                "EligibilityCriteria",
                "MinimumAge",
                "MaximumAge",
                "Sex",
                "HealthyVolunteers",
                "LeadSponsorName",
                "LastUpdatePostDate",
                "EnrollmentCount",
                "LocationFacility",
                "LocationCity",
                "LocationState",
                "LocationCountry",
                "LocationStatus",
                "CentralContactName",
                "CentralContactRole",
                "CentralContactEMail",
            ]
        ),
    }

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; ctgov-downloader/1.0)"})
    while True:
        params = dict(base_params)
        if state["next_page_token"]:
            params["pageToken"] = state["next_page_token"]

        print(
            f"Fetching page {state['page'] + 1}"
            f" (so far {state['total_studies']} studies)",
            file=sys.stderr,
        )
        data = fetch_page(session, params)

        studies = data.get("studies", [])
        if not studies:
            print("No more studies — done.")
            break

        out_file = args.output / f"page_{state['page']:05d}.json"
        out_file.write_text(json.dumps(studies, ensure_ascii=False), encoding="utf-8")
        state["total_studies"] += len(studies)
        state["page"] += 1

        state["next_page_token"] = data.get("nextPageToken")
        state_path.write_text(json.dumps(state), encoding="utf-8")

        if not state["next_page_token"]:
            print("No nextPageToken — finished.")
            break
        if args.max_pages and state["page"] >= args.max_pages:
            print(f"Reached max-pages={args.max_pages}, stopping.")
            break

    print(f"Done. Downloaded {state['total_studies']} studies into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
