
from pathlib import Path
from datetime import datetime
import requests
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FILE = PROJECT_ROOT / "data" / "public_holidays.csv"

SOURCE_URL = (
    "https://raw.githubusercontent.com/"
    "ppraserts/thailand-open-data/main/data/"
    "thai-public-holidays/all.json"
)


def download_holidays():

    print("Downloading Thai public holiday data...")

    try:
        response = requests.get(
            SOURCE_URL,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

    except requests.exceptions.RequestException as error:
        print(f"Download failed: {error}")
        return None

    except ValueError:
        print("The downloaded data is not valid JSON.")
        return None

    if "years" not in data:
        print("Unexpected JSON format: missing 'years'.")
        return None

    return data


def process_holidays(data):

    records = []

    for year_data in data["years"]:

        year = year_data.get("year")
        buddhist_year = year_data.get("year_buddhist_era")

        for holiday in year_data.get("holidays", []):

            date_value = holiday.get("date")

            try:
                date = datetime.strptime(
                    date_value,
                    "%Y-%m-%d"
                )

            except (ValueError, TypeError):
                print(f"Skipping invalid date: {date_value}")
                continue

            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "year": date.year,
                "buddhist_year": buddhist_year,
                "weekday": date.strftime("%A"),
                "name_th": holiday.get("name_th") or "",
                "name_en": holiday.get("name_en") or "",
                "holiday_type": holiday.get("type") or "",
                "is_substitution": bool(
                    holiday.get("is_substitution", False)
                ),
                "note": holiday.get("note") or "",
                "source": data.get("source", ""),
                "source_url": SOURCE_URL
            })

    if not records:
        print("No holiday records were found.")
        return None

    df = pd.DataFrame(records)

    # Remove exact duplicate holiday records
    df = df.drop_duplicates(
        subset=["date", "name_th", "name_en", "holiday_type"]
    )

    # Sort chronologically
    df = df.sort_values(
        by=["date", "name_en"]
    ).reset_index(drop=True)

    return df


def export_data(df):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nCSV successfully created.")
    print(f"Output: {OUTPUT_FILE}")


def main():

    data = download_holidays()

    if data is None:
        return

    print(f"Source: {data.get('source', 'Unknown')}")
    print(f"Available years: {data.get('available_years', [])}")

    df = process_holidays(data)

    if df is None:
        return

    export_data(df)

    print("------------------")
    print(f"Total holidays: {len(df)}")
    print(f"Years included: {sorted(df['year'].unique())}")

    print("\nHoliday types:")
    print(df["holiday_type"].value_counts().to_string())

    print("\nFirst five records:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
