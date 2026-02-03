import pandas as pd
import requests
import time

# Set user agent
headers = {"User-Agent": "laksolajoni@gmail.com"}

# Get company tickers
companyTickers = requests.get(
    "https://www.sec.gov/files/company_tickers.json", headers=headers
)

# Let's investigate the json structure
print(companyTickers.json().keys())
print(companyTickers.json())
companyTickers.json()["0"]

# Convert to DataFrame
companyData = pd.DataFrame.from_dict(companyTickers.json(), orient="index")

# Add leading zeros to CIK
companyData["cik_str"] = companyData["cik_str"].astype(str).str.zfill(10)
print(companyData.head())
print(len(companyData))

# 3674 is the SIC code for Semiconductors and Related Devices
# Let's now create a list of companies' tickers probably in that industry
tickers_3674 = [
    "AMD",
    "NVDA",
    "AVGO",
    "MRVL",
    "ON",
    "QRVO",
    "SWKS",
    "NXPI",
    "CRUS",
    "SIMO",
    "HIMX",
    "MXL",
    "SMTC",
    "SLAB",
    "SYNA",
    "RMBS",
    "POWI",
    "MPWR",
    "LSCC",
    "DIOD",
    "AOSL",
    "AMBA",
    "CRDO",
    "MTSI",
    "QUIK",
    "GSIT",
    "MRAM",
    "NLST",
    "SITM",
    "ALGM",
    "ALMU",
    "LAES",
    "POET",
    "PXLW",
    "KOPN",
    "LPTH",
    "AIP",
    "ATOM",
    "AXTI",
    "ICHR",
    "UCTT",
    "FORM",
    "PLAB",
    "KLIC",
    "AMKR",
    "OSIS",
    "SKYT",
    "GFS",
    "WOLF",
    "INDI",
]

seed = (companyData[companyData["ticker"].isin(tickers_3674)]).copy()
print(len(seed))


seed["sic"] = ""
seed["sicDescription"] = ""
all_rows = len(seed)
round_number = 0

for index, row in seed.iterrows():
    cik = row["cik_str"]
    companySubmissions = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json", headers=headers
    )

    # get the SIC code and description
    sic_code = companySubmissions.json()["sic"]
    sicDescription = companySubmissions.json()["sicDescription"]
    # add to dataframe
    seed.at[index, "sic"] = sic_code
    seed.at[index, "sicDescription"] = sicDescription
    # print progress
    round_number += 1
    print(
        f"Processed row {round_number} from {all_rows} with CIK {cik} and SIC {sic_code}"
    )
    # sleep to avoid rate limiting
    time.sleep(0.2)

print(seed.head())
print(seed["sic"].unique())

# Save to CSV
seed.to_csv(
    "/Users/kayttaja/Desktop/PREDICTING EARNINGS/data/interim/semiconductor_companies.csv",
    index=False,
)
