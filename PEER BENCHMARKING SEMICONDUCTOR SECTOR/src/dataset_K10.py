import time
import requests
import pandas as pd

# Set user agent
headers = {"User-Agent": "laksolajoni@gmail.com"}

df_CIK = pd.read_csv(
    "/Users/kayttaja/Desktop/PEER BENCHMARKING SEMICONDUCTOR SECTOR/data/interim/semiconductor_companies.csv",
    dtype={"cik_str": str},
)


# Function to extract fact data
def extract_fact(facts_json, fact_name):
    # Try both US GAAP and IFRS tags
    standard_tags = ["us-gaap", "ifrs-full"]
    # Extract fact data
    for standard_tag in standard_tags:
        try:
            fact_data = facts_json["facts"][standard_tag][fact_name]["units"]["USD"]
            break
        except KeyError:
            continue
    else:
        return pd.DataFrame(
            columns=["accn", "end", "fp", "form", "filed", fact_name.lower()]
        )
    # Convert to DataFrame
    df_fact = pd.DataFrame(fact_data)
    # Filter for 10-K forms
    df_fact = df_fact[df_fact["form"].isin(["10-K", "20-F/A", "20-F"])].copy()
    # Convert 'filed' to datetime and sort
    if "filed" in df_fact.columns:
        df_fact["filed"] = pd.to_datetime(df_fact["filed"], errors="coerce")
        df_fact = df_fact.sort_values("filed")

    # Remove duplicates based on 'end', 'fp' and 'form'
    df_fact = df_fact.drop_duplicates(subset=["end", "fp", "form"], keep="last")

    # Rename 'val' column to fact name in lowercase
    df_fact = df_fact.rename(columns={"val": fact_name.lower()})
    return df_fact


def extract_fact_with_fallback(facts_json, tags, output_name):
    frames = []
    for tag in tags:
        df_fact = extract_fact(facts_json, tag)
        if df_fact.empty:
            continue
        if tag.lower() in df_fact.columns and output_name not in df_fact.columns:
            df_fact = df_fact.rename(columns={tag.lower(): output_name})
        frames.append(df_fact)

    if not frames:
        return pd.DataFrame(columns=["accn", "end", "fp", "form", "filed", output_name])

    # Combine all DataFrames, prioritizing earlier tags
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["accn", "end"])
    df = df.drop_duplicates(subset=["accn", "end"], keep="last")
    return df


# List of facts to extract with fallback tags for each fact
Fallback_tags = {
    "Revenues": [
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenue",
    ],
    "ResearchAndDevelopmentExpense": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
    ],
    "Assets": ["Assets", "AssetsCurrent"],
    "Liabilities": ["Liabilities", "LiabilitiesCurrent"],
    "CashAndCashEquivalentsAtCarryingValue": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "InventoryNet": ["InventoryNet", "Inventories"],
    "AccountsReceivableNetCurrent": [
        "AccountsReceivableNetCurrent",
        "AccountsReceivableNet",
    ],
    "AccountsPayableCurrent": [
        "AccountsPayableCurrent",
        "AccountsPayable",
        "AccountsPayableTradeCurrent",
        "AccountsPayableTrade",
    ],
    "PropertyPlantAndEquipmentNet": [
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentNetIncludingConstructionInProgress",
    ],
    "NetCashProvidedByUsedInOperatingActivities": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        "NetCashProvidedByUsedInOperatingActivitiesDiscontinuedOperations",
    ],
    "SellingGeneralAndAdministrativeExpense": [
        "SellingGeneralAndAdministrativeExpense"
    ],
    "SellingAndMarketingExpense": ["SellingAndMarketingExpense"],
    "GeneralAndAdministrativeExpense": ["GeneralAndAdministrativeExpense"],
    "GrossProfit": ["GrossProfit"],
    "OperatingExpenses": ["OperatingExpenses"],
    "CostOfRevenue": ["CostOfRevenue"],
    "CostOfGoodsAndServicesSold": ["CostOfGoodsAndServicesSold"],
    "PropertyPlantAndEquipmentGross": ["PropertyPlantAndEquipmentGross"],
    "AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment": [
        "AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment"
    ],
}

all_parts = []

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
    for fact in Fallback_tags.keys():
        if fact == "Revenues":
            continue
        fact_dfs[fact] = extract_fact_with_fallback(
            facts_json, Fallback_tags[fact], fact.lower()
        )

    # Merge all fact DataFrames on join keys
    join_keys = ["accn"]
    df_merged = extract_fact_with_fallback(
        facts_json, Fallback_tags["Revenues"], "revenue"
    )
    # Skip merge if left DataFrame is empty or missing join keys
    if df_merged.empty or not set(join_keys).issubset(df_merged.columns):
        print(f"Skip {ticker}: no revenue rows")
        continue

    for fact in Fallback_tags.keys():
        if fact == "Revenues":
            continue
        df_right = fact_dfs[fact]
        needed_cols = set(join_keys + [fact.lower()])
        # Skip merge if right DataFrame is empty or missing needed columns
        if df_right.empty or not needed_cols.issubset(df_right.columns):
            continue
        # Otherwise, perform the merge
        df_right = df_right[join_keys + [fact.lower()]]
        df_merged = pd.merge(df_merged, df_right, on=join_keys, how="left")

    # Add company info
    df_merged["company_name"] = company_name
    df_merged["ticker"] = ticker
    df_merged["cik"] = cik

    # Append to all parts list
    all_parts.append(df_merged)

    # Print progress and sleep to avoid rate limiting
    print(f"Processed {company_name}")
    time.sleep(0.2)


# Concatenate all company DataFrames
df = pd.concat(all_parts, ignore_index=True)

# Convert 'end' and 'start' to datetime and calculate duration in days
df["start"] = pd.to_datetime(df["start"], errors="coerce")
df["end"] = pd.to_datetime(df["end"], errors="coerce")
df["days"] = (df["end"] - df["start"]).dt.days
# Filter for rows where duration is between 330 and 400 days
df = df[df["days"].between(330, 400)]

# Check for missing values and print the number of rows
print(df.isna().sum())
print(len(df))

# Count rows where revenue is zero
revenue_zero_count = 0
for _, row in df.iterrows():
    if row["revenue"] == 0:
        revenue_zero_count += 1
print(f"Number of rows with zero revenue: {revenue_zero_count}")


# Save to CSV
df.to_csv(
    "/Users/kayttaja/Desktop/PEER BENCHMARKING SEMICONDUCTOR SECTOR/data/interim/semiconductor_companies_10K.csv",
    index=False,
)
