import time
import requests
import pandas as pd

# Set user agent
headers = {"User-Agent": "laksolajoni@gmail.com"}

df_CIK = pd.read_csv(
    "/Users/kayttaja/Desktop/PREDICTING EARNINGS/data/interim/semiconductor_companies.csv",
    dtype={"cik_str": str},
)
print(df_CIK.head(10))
print(df_CIK["sic"].unique())
print(df_CIK.isna().sum())

df = pd.DataFrame(
    {
        "company_name": [],
        "ticker": [],
        "cik": [],
        "revenue": [],
        "rd_expense": [],
        "operating_income": [],
        "fiscal_year": [],
        "fiscal_period": [],
        "frame": [],
        "accn": [],
        "sga_expense": [],
        "period_end_date": [],
        "period_start_date": [],
        "filing_date": [],
        "form": [],
        "gross_profit": [],
        "operating_expense": [],
        "total_assets": [],
        "total_liabilities": [],
        "cash_and_equivalents": [],
        "inventory": [],
        "accounts_receivable": [],
        "accounts_payable": [],
        "ppe_net": [],
        "operating_cash_flow": [],
        "capex": [],
        "free_cash_flow": [],
    }
)

# Set up empty DataFrame to collect all data
df = pd.DataFrame()


# Function to extract fact data
def extract_fact(facts_json, fact_name):
    try:
        # Extract fact data
        fact_data = facts_json["facts"]["us-gaap"][fact_name]["units"]["USD"]
        # Convert to DataFrame
        df_fact = pd.DataFrame(fact_data)
        # Filter for 10-Q and 10-K forms
        df_fact = df_fact[df_fact["form"].isin(["10-Q", "10-K"])].copy()

        # Convert 'filed' to datetime and sort
        if "filed" in df_fact.columns:
            df_fact["filed"] = pd.to_datetime(df_fact["filed"], errors="coerce")
            df_fact = df_fact.sort_values("filed")
        # Remove duplicates based on 'end' and 'fp'
        if "fp" in df_fact.columns:
            df_fact = df_fact.drop_duplicates(subset=["end", "fp"], keep="last")
        # Rename 'val' column to fact name in lowercase
        df_fact = df_fact.rename(columns={"val": fact_name.lower()})
        return df_fact
    except KeyError:
        return pd.DataFrame()


def extract_fact_with_fallback(facts_json, tags, output_name):
    for tag in tags:
        df_tag = extract_fact(facts_json, tag)
        if df_tag.empty:
            continue
        if tag.lower() in df_tag.columns and output_name not in df_tag.columns:
            df_tag = df_tag.rename(columns={tag.lower(): output_name})
        return df_tag
    return pd.DataFrame(columns=["end", "fp", output_name])


# List of facts to extract
fact_list = [
    "ResearchAndDevelopmentExpense",
    "OperatingIncomeLoss",
    "SellingGeneralAndAdministrativeExpense",
    "GrossProfit",
    "OperatingExpenses",
    "Assets",
    "Liabilities",
    "CashAndCashEquivalentsAtCarryingValue",
    "InventoryNet",
    "AccountsReceivableNetCurrent",
    "AccountsPayableCurrent",
    "PropertyPlantAndEquipmentNet",
    "NetCashProvidedByUsedInOperatingActivities",
]

REVENUE_TAGS = [
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
]

# Loop through each company CIK
for _, row in df_CIK.iterrows():
    # Get CIK, company name, and ticker
    cik = row["cik_str"]
    company_name = row["title"]
    ticker = row["ticker"]

    # Get company facts
    facts_json = requests.get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=headers
    ).json()

    # Dictionary to hold DataFrames for each fact
    fact_dfs = {}

    # Extract each fact and store in the dictionary
    for fact in fact_list:
        fact_dfs[fact] = extract_fact(facts_json, fact)

    # Merge all fact DataFrames on join keys
    join_keys = ["end", "fp"]
    df_merged = extract_fact_with_fallback(facts_json, REVENUE_TAGS, "revenue")
    for fact in fact_list:
        df_right = fact_dfs[fact]
        needed_cols = set(join_keys + [fact.lower()])
        # Skip merge if right DataFrame is empty or missing needed columns
        if df_right.empty or not needed_cols.issubset(df_right.columns):
            continue
        # Otherwise, perform the merge
        df_right = df_right[join_keys + [fact.lower()]]
        df_merged = pd.merge(df_merged, df_right, on=join_keys, how="left")
        if df_merged.empty or not set(join_keys).issubset(df_merged.columns):
            print(f"Skip {ticker}: no revenue rows")
            continue

    # Add company info
    df_merged["company_name"] = company_name
    df_merged["ticker"] = ticker
    df_merged["cik"] = cik

    # Append to main DataFrame
    df = pd.concat([df, df_merged], ignore_index=True)

    # Print progress and sleep to avoid rate limiting
    print(f"Processed {company_name}")
    time.sleep(0.2)


print(df.isna().sum())
print(len(df))

# Save to CSV
df.to_csv(
    "/Users/kayttaja/Desktop/PREDICTING EARNINGS/data/interim/semiconductor_companies_10Q_10K_facts.csv",
    index=False,
)
