import pandas as pd
import requests
import matplotlib.pyplot as plt

headers = {"User-Agent": "laksolajoni@gmail.com"}

companyTickers = requests.get(
    "https://www.sec.gov/files/company_tickers.json", headers=headers
)

print(companyTickers.json().keys())

firstEntry = companyTickers.json()["0"]
print(firstEntry)

directCIK = companyTickers.json()["0"]["cik_str"]
print(directCIK)

companyData = pd.DataFrame.from_dict(companyTickers.json(), orient="index")
print(companyData.head())

# add leading zeros to CIK
companyData["cik_str"] = companyData["cik_str"].astype(str).str.zfill(10)
print(companyData.head())

cik = companyData["cik_str"].iloc[0]
print(cik)

filingMetadata = requests.get(
    f"https://data.sec.gov/submissions/CIK{cik}.json", headers=headers
)

# review json
print(filingMetadata.json().keys())
filingMetadata.json()["filings"]
filingMetadata.json()["filings"].keys()
filingMetadata.json()["filings"]["recent"]
filingMetadata.json()["filings"]["recent"].keys()

# dictionary to dataframe
allForms = pd.DataFrame.from_dict(filingMetadata.json()["filings"]["recent"])

print(allForms.head())
print(allForms.columns)

# get company facts
companyFacts = requests.get(
    f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=headers
)

# review data
companyFacts.json().keys()
companyFacts.json()["facts"]
companyFacts.json()["facts"].keys()
companyFacts.json()["facts"]["dei"]
companyFacts.json()["facts"]["dei"].keys()

# filing metadata
companyFacts.json()["facts"]["dei"]["EntityCommonStockSharesOutstanding"]
companyFacts.json()["facts"]["dei"]["EntityCommonStockSharesOutstanding"].keys()
companyFacts.json()["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]
companyFacts.json()["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"][
    "shares"
]
companyFacts.json()["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"][
    "shares"
][0]

# concept data // financial statement line items
companyFacts.json()["facts"]["us-gaap"]
companyFacts.json()["facts"]["us-gaap"].keys()

# different amounts of data available per concept
companyFacts.json()["facts"]["us-gaap"]["AccountsPayable"]
companyFacts.json()["facts"]["us-gaap"]["Revenues"]
companyFacts.json()["facts"]["us-gaap"]["Assets"]
companyFacts.json()["facts"]["us-gaap"]["Assets"]["units"]["USD"][0]["val"]
companyFacts.json()["facts"]["us-gaap"]["ResearchAndDevelopmentExpense"]

companyConcept = requests.get(
    (f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/Assets.json"),
    headers=headers,
)

# review data
companyConcept.json().keys()
companyConcept.json()["units"]
companyConcept.json()["units"].keys()
companyConcept.json()["units"]["USD"]
companyConcept.json()["units"]["USD"][0]

# parse assets from single filing
companyConcept.json()["units"]["USD"][0]["val"]

# get all filings data
assetsData = pd.DataFrame.from_dict((companyConcept.json()["units"]["USD"]))

# review data
assetsData.columns
assetsData.form

# get assets from 10Q forms and reset index
assets10Q = assetsData[assetsData.form == "10-Q"]
assets10Q = assets10Q.reset_index(drop=True)

assets10Q.head()

# plot
assets10Q.plot(x="end", y="val")

# Let's check the submissions section
companySubmissions = requests.get(
    f"https://data.sec.gov/submissions/CIK{cik}.json", headers=headers
)

# review data
companySubmissions.json().keys()
companySubmissions.json()["sicDescription"]
companySubmissions.json()["sic"]
companySubmissions.json()["name"]
companySubmissions.json()["cik"]
companySubmissions.json()["filings"].keys()
companySubmissions.json()["filings"]["recent"].keys()
companySubmissions.json()["filings"]["recent"]["form"]
companySubmissions.json()["filings"]["recent"]["primaryDocument"]
companySubmissions.json()["filings"]["recent"]["accessionNumber"]

# let's make a dataframe of form, primaryDocument, accessionNumber
recentFilings = pd.DataFrame.from_dict(companySubmissions.json()["filings"]["recent"])[
    ["form", "primaryDocument", "accessionNumber"]
]
print(recentFilings.head())

tenk = recentFilings[recentFilings["form"] == "10-K"].copy()
print(tenk.head())

cik_str = "0001045810"  # tai seedistäsi
cik_no_zeros = str(int(cik_str))  # "1045810"

tenk = tenk.sort_values("accessionNumber", ascending=False).iloc[0]  # uusin 10-K
acc = tenk["accessionNumber"].replace("-", "")  # poista viivat
doc = tenk["primaryDocument"]

url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{acc}/{doc}"
print(url)

from bs4 import BeautifulSoup
import re

r = requests.get(url, headers=headers, timeout=30)
soup = BeautifulSoup(r.text, "html.parser")
text = soup.get_text(" ", strip=True)

# etsi sanan "fabless" ympäriltä katkelma
m = re.search(r"(?i).{0,30}\bfabless\b.{0,30}", text)
print(m.group(0) if m else "Ei löytynyt 'fabless' sanaa tästä dokumentista.")


cik_str = "0001045810"

facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str}.json"
facts = requests.get(facts_url, headers=headers).json()
print(facts.keys())
facts["entityName"]
list(facts["facts"].keys())
facts["facts"]["us-gaap"].keys()
facts["facts"]["us-gaap"]["CapitalExpenditures"]
facts["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"]
facts["facts"]["us-gaap"]["ResearchAndDevelopmentExpense"]
fact_data = facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
fact_data = pd.DataFrame(fact_data)
fact_data = fact_data[fact_data["form"].isin(["10-K", "20-F/A"])].copy()
print(fact_data["form"].unique())


revenues = (
    facts.get("facts", {})
    .get("ifrs-full", {})
    .get("Revenues", {})
    .get("units", {})
    .get("USD", [])
)
print(revenues)

df = pd.DataFrame(revenues)
print(df.head())
print(len(df))
df = df[df["form"].isin(["10-K", "20-F/A"])].rename(columns={"val": "revenue"}).copy()
print(df.head())
print(len(df))


facts["facts"]["ifrs-full"]["OperatingIncomeLoss"]["units"]["USD"][0]
operating_income = (
    facts.get("facts", {})
    .get("ifrs-full", {})
    .get("OperatingIncomeLoss", {})
    .get("units", {})
    .get("USD", [])
)

df_operating_income = pd.DataFrame(operating_income)
df_operating_income = (
    df_operating_income[df_operating_income["form"].isin(["10-K", "20-F/A"])]
    .rename(columns={"val": "operating_income"})
    .copy()
)
print(df_operating_income.head())


# merge revenues and operating income
join_keys = ["end", "form", "frame", "accn"]
df_right = df_operating_income[join_keys + ["operating_income"]]
df_merged = pd.merge(df, df_right, on=join_keys, how="left")

