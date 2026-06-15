import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


load_dotenv()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
HEADERS = {"Accept": "application/vnd.github.v3+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

SEARCH_GROUPS = [
    "followers:1..20",
    "followers:21..100",
    "followers:101..1000",
    "followers:>1000",
]
SEARCH_PER_PAGE = 100
SEARCH_MAX_PAGES = 3
SEARCH_ENDPOINT = "https://api.github.com/search/users"
USER_ENDPOINT = "https://api.github.com/users/{username}"

# A list of real public GitHub usernames to fetch by default.
# This is kept as a stable fallback and for offline/demo use.
def fetch_usernames_via_search(n=700):
    """Fetch n real GitHub usernames sorted by followers."""
    usernames = []
    page = 1
    while len(usernames) < n:
        r = requests.get(
            "https://api.github.com/search/users",
            params={"q": "followers:>50", "sort": "followers", "order": "desc",
                    "per_page": 100, "page": page},
            headers=HEADERS, timeout=10
        )
        items = r.json().get("items", [])
        if not items:
            break
        usernames.extend(u["login"] for u in items)
        page += 1
        if page > 10:
            break
        time.sleep(0.5)
    return list(dict.fromkeys(usernames))[:n]


GITHUB_USERNAMES = fetch_usernames_via_search(700)

GITHUB_USERNAMES = list(dict.fromkeys(GITHUB_USERNAMES))


def create_session() -> requests.Session:
    """Create a requests session with retry support."""
    session = requests.Session()
    session.headers.update(HEADERS)

    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _safe_get(session: requests.Session, url: str, params: dict | None = None, timeout: int = 10):
    """Perform a GET request and handle rate limit retries."""
    response = session.get(url, params=params, timeout=timeout)
    if response.status_code == 403 and "rate limit" in response.text.lower():
        wait = int(response.headers.get("Retry-After", 60))
        print(f"Rate limited, sleeping {wait}s before retrying")
        time.sleep(wait)
        response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response


def search_github_usernames(
    search_groups: list[str] = SEARCH_GROUPS,
    per_page: int = SEARCH_PER_PAGE,
    max_pages: int = SEARCH_MAX_PAGES,
) -> list[str]:
    """Discover GitHub usernames using the GitHub Search API."""
    session = create_session()
    usernames: list[str] = []

    for group in search_groups:
        print(f"Searching group: {group}")
        for page in range(1, max_pages + 1):
            params = {
                "q": group,
                "per_page": per_page,
                "page": page,
            }
            try:
                response = _safe_get(session, SEARCH_ENDPOINT, params=params)
            except requests.HTTPError as error:
                print(f"  Search request failed for {group} page {page}: {error}")
                break

            items = response.json().get("items", [])
            if not items:
                break

            page_usernames = [item["login"] for item in items if item.get("login")]
            print(f"  Page {page}: found {len(page_usernames)} users")
            usernames.extend(page_usernames)
            if len(items) < per_page:
                break
            time.sleep(1)

    return list(dict.fromkeys(usernames))


def fetch_github_users(usernames: list[str]) -> pd.DataFrame:
    """Fetch raw profile data for a list of GitHub usernames."""
    session = create_session()
    records: list[dict] = []
    print(f"Fetching {len(usernames)} GitHub user profiles...")

    for index, username in enumerate(usernames, start=1):
        try:
            response = _safe_get(session, USER_ENDPOINT.format(username=username))
            data = response.json()
            records.append({
                "username": username,
                "public_repos": data.get("public_repos", 0),
                "public_gists": data.get("public_gists", 0),
                "followers": data.get("followers", 0),
                "following": data.get("following", 0),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "hireable": data.get("hireable"),
                "site_admin": data.get("site_admin", False),
            })
            print(f"  [{index}/{len(usernames)}] {username}: OK")
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else "?"
            print(f"  [{index}/{len(usernames)}] {username}: HTTP {status} - {error}")
        except Exception as error:
            print(f"  [{index}/{len(usernames)}] {username}: error - {error}")

        time.sleep(0.5)

    columns = [
        "username",
        "public_repos",
        "public_gists",
        "followers",
        "following",
        "created_at",
        "updated_at",
        "hireable",
        "site_admin",
    ]
    df = pd.DataFrame(records, columns=columns)
    print(f"\nFetched {len(df)} users successfully.")
    return df


def fetch_github_users_from_search_groups(
    search_groups: list[str] = SEARCH_GROUPS,
    per_page: int = SEARCH_PER_PAGE,
    max_pages: int = SEARCH_MAX_PAGES,
) -> pd.DataFrame:
    """Search GitHub users by follower groups and fetch profile data."""
    usernames = search_github_usernames(search_groups, per_page, max_pages)
    if not usernames:
        print("WARNING: no usernames found from search groups; using fallback username list.")
        usernames = GITHUB_USERNAMES
    print(f"Found {len(usernames)} unique usernames from search groups.")
    return fetch_github_users(usernames)


def label_churn(df: pd.DataFrame, days_threshold: int = 180) -> pd.DataFrame:
    """
    Define churn: a user is 'churned' if their last activity
    (updated_at) was more than `days_threshold` days ago.
    180 days is a reasonable default for GitHub — developers
    can go quiet for months during project cycles.
    """
    df = df.copy()
    if df.empty:
        raise ValueError("Received empty DataFrame from GitHub fetcher; check API connectivity and token settings.")
    if "updated_at" not in df.columns:
        raise ValueError("Missing required column 'updated_at' in fetched data.")

    df["last_active"] = pd.to_datetime(df["updated_at"], utc=True)
    now = datetime.now(timezone.utc)

    df["days_inactive"] = (now - df["last_active"]).dt.days
    df["churned"] = (df["days_inactive"] > days_threshold).astype(int)

    churned_pct = df["churned"].mean() * 100
    print(f"Churn label: {days_threshold} days threshold")
    print(f"Churned: {df['churned'].sum()} / {len(df)} ({churned_pct:.1f}%)")
    return df


if __name__ == "__main__":
    os.makedirs("../data/raw", exist_ok=True)
    df_raw = fetch_github_users_from_search_groups()
    df_raw.to_csv("../data/raw/github_users_raw.csv", index=False)
    df_labeled = label_churn(df_raw)
    df_labeled.to_csv("../data/raw/github_users_labeled.csv", index=False)
    print("Saved to data/raw/")