"""
Merge all_articles_classified.csv with 230126_Medienliste.xlsx
Matches on domain (extracted from Excel URL column).
Adds all Excel columns EXCEPT: Kanal, Sprache, Land.
"""

import pandas as pd
from urllib.parse import urlparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR.parent / "classification_pipeline" / "euroBERT_210m" / "local_data" / "all_articles_classified.csv"
EXCEL_PATH = BASE_DIR / "data" / "230126_Medienliste.xlsx"
OUTPUT_PATH = BASE_DIR / "data" / "articles_enriched.csv"

EXCLUDE_COLS = {"Kanal", "Sprache", "Land"}


def extract_domain(url):
    if pd.isna(url):
        return None
    try:
        parsed = urlparse(str(url))
        host = parsed.netloc or parsed.path
        host = host.replace("www.", "").rstrip("/")
        return host.lower()
    except Exception:
        return None


def main():
    # Load data
    csv_df = pd.read_csv(CSV_PATH, low_memory=False)
    excel_df = pd.read_excel(EXCEL_PATH)

    # Extract domain from Excel URLs
    excel_df["domain"] = excel_df["URL"].apply(extract_domain)

    # Drop excluded columns and the raw URL column
    drop_cols = [c for c in excel_df.columns if c in EXCLUDE_COLS or c == "URL"]
    excel_df = excel_df.drop(columns=drop_cols)

    # Deduplicate: keep first entry per domain
    excel_dedup = excel_df.dropna(subset=["domain"]).drop_duplicates(subset="domain", keep="first")

    # Normalize CSV domain for matching
    csv_df["domain_lower"] = csv_df["domain"].str.lower().str.strip()
    excel_dedup = excel_dedup.copy()
    excel_dedup["domain_lower"] = excel_dedup["domain"].str.lower().str.strip()

    # Merge
    merged = csv_df.merge(
        excel_dedup.drop(columns=["domain"]),
        on="domain_lower",
        how="left",
    )
    merged = merged.drop(columns=["domain_lower"])

    # Report
    n_matched = merged[merged["Publikation"].notna()]["domain"].nunique()
    n_total = merged["domain"].nunique()
    print(f"Domains matched: {n_matched}/{n_total}")
    unmatched = merged[merged["Publikation"].isna()]["domain"].unique()
    if len(unmatched) > 0:
        print(f"Unmatched domains: {list(unmatched)}")

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved enriched data to {OUTPUT_PATH} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
