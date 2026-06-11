import requests
import pandas as pd
import time
from datetime import datetime, timezone


# A list of real public GitHub usernames to fetch
GITHUB_USERNAMES = [
    "torvalds", "gvanrossum", "antirez", "mitchellh", "fabpot",
    "yyx990803", "tj", "sindresorhus", "mrdoob", "addyosmani",
    "nicowillis", "jashkenas", "jeresig", "Marak", "paulirish",
    "substack", "isaacs", "jlong", "fat", "mbostock",
    "tenderlove", "defunkt", "pjhyett", "wycats", "ezmobius",
    "atmos", "KirinDave", "jamesgolick", "brynary", "technoweenie",
    "schacon", "rtomayko", "mojombo", "bmizerany", "caged",
    "hornbeck", "nex3", "jtauber", "drnic", "remi",
    "dkubb", "radar", "bkeepers", "jnunemaker", "al3x",
    "joshsusser", "danwrong", "harthur", "ChrisKempson", "gelstudios",
    "poteto", "zenorocha", "jakubroztocil", "tlrobinson", "paulmillr",
    "florinpop17", "kamranahmedse", "bradtraversy", "wesbos", "tannerlinsley",
    "gaearon", "acdlite", "sebmarkbage", "zpao", "spicyj",
    "necolas", "fat", "mdo", "dhh", "rails",
    "matz", "yukihiro-matz", "tenderlove", "evanphx", "burke",
    "kennethreitz", "armon", "mitchellh", "hashicorp", "nicowillis",
    "marionebl", "nicolo-ribaudo", "nickmccurdy", "nicolo-ribaudo", "babel",
    "facebook", "google", "microsoft", "apple", "twitter",
    "jdalton", "nicolo-ribaudo", "nickmccurdy", "ljharb", "mathiasbynens",
    "davidfowl", "shanselman", "migueldeicaza", "mono", "dotnet",
    "antirez", "pietern", "mattn", "BurntSushi", "ogham",
    "mozilla", "servo", "rust-lang", "golang", "python",
]

# Remove duplicates
GITHUB_USERNAMES = list(dict.fromkeys(GITHUB_USERNAMES))


def fetch_github_users(usernames: list) -> pd.DataFrame:
    """Fetch raw data for a list of GitHub usernames."""
    records = []
    print(f"Fetching {len(usernames)} GitHub users...")

    for i, username in enumerate(usernames):
        try:
            r = requests.get(
                f"https://api.github.com/users/{username}",
                headers={"Accept": "application/vnd.github.v3+json"},
                timeout=10
            )
            if r.status_code == 404:
                print(f"  [{i+1}] {username}: not found, skipping")
                continue
            if r.status_code == 403:
                print(f"  [{i+1}] Rate limit hit, waiting 60s...")
                time.sleep(60)
                r = requests.get(f"https://api.github.com/users/{username}", timeout=10)

            data = r.json()
            records.append({
                "username":     username,
                "public_repos": data.get("public_repos", 0),
                "public_gists": data.get("public_gists", 0),
                "followers":    data.get("followers", 0),
                "following":    data.get("following", 0),
                "created_at":   data.get("created_at"),
                "updated_at":   data.get("updated_at"),
                "hireable":     data.get("hireable"),
                "site_admin":   data.get("site_admin", False),
            })
            print(f"  [{i+1}] {username}: OK")
        except Exception as e:
            print(f"  [{i+1}] {username}: error - {e}")

        time.sleep(0.5)  # Respect rate limits (60 req/hour unauthenticated)

    df = pd.DataFrame(records)
    print(f"\nFetched {len(df)} users successfully.")
    return df


def label_churn(df: pd.DataFrame, days_threshold: int = 180) -> pd.DataFrame:
    """
    Define churn: a user is 'churned' if their last activity
    (updated_at) was more than `days_threshold` days ago.
    180 days is a reasonable default for GitHub — developers
    can go quiet for months during project cycles.
    """
    df = df.copy()
    df["last_active"] = pd.to_datetime(df["updated_at"], utc=True)
    now = datetime.now(timezone.utc)

    df["days_inactive"] = (now - df["last_active"]).dt.days
    df["churned"] = (df["days_inactive"] > days_threshold).astype(int)

    churned_pct = df["churned"].mean() * 100
    print(f"Churn label: {days_threshold} days threshold")
    print(f"Churned: {df['churned'].sum()} / {len(df)} ({churned_pct:.1f}%)")
    return df


if __name__ == "__main__":
    df_raw = fetch_github_users(GITHUB_USERNAMES)
    df_raw.to_csv("../data/raw/github_users_raw.csv", index=False)
    df_labeled = label_churn(df_raw)
    df_labeled.to_csv("../data/raw/github_users_labeled.csv", index=False)
    print("Saved to data/raw/")
