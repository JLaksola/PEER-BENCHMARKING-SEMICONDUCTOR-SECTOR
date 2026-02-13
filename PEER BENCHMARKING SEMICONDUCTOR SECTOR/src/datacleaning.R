# ----
# Downloading the required packages
required_packages <- c(
  "tidyverse",  
  "lubridate",  
  "caret",      
  "corrplot",   
  "pROC",
  "ts",
  "ggplot2",
  "flextable",
  "readr"
)

# Check if any package needs to be installed, install what is needed
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

# Load the required packages
lapply(required_packages, library, character.only = TRUE)
# -----


# Import the data
df_smc <- read_csv("semiconductor_companies_10k.csv")
summary(df_smc)



# -------------------------------------------------
# DATA PREPARATION
# -------------------------------------------------

# Cleaning the Dates & Createing Year-column
# The 'end' column tells us which fiscal year the data belongs to.
df_cleaning <- df_smc %>%
  mutate(
    date = ymd(end),  # Convert text to Date format
    year = year(date) # Extract the year for grouping
  )

# Selecting meaningful columns, and creating target variable columns:
# Remove AMORTIZATION OF INTANGIBLE ASSETS, FRAME, FORM, FY, FP and ACCN
    # -> They held a large amount of NA's, and/or have no meaningful effect on EBIT Margins
# Add ebit_margin, rnd_per_revenue, capex_margin, capex_per_assets, asset_turnover, intangible_intensity, revenue_growth,
      # current_asset_ratio
# Recalculate "intangible assets net including goodwill" as "intangible assets net excluding goodwill" + goodwill
# COLUMNS FOR OUTLIER TREATMENT: size_log

df_cleaning <- df_cleaning %>%
  
  #REMOVING THE UNNECESSARY COLUMNS
  select(-intangibleassetsnetincludinggoodwill,
         -amortizationofintangibleassets,
         -frame,
         -form,
         -fy,
         -fp,
         -accn) %>% 
  
  # ADDING NEW COLUMNS FOR PREDICTING EBITMARGIN PERFORMANCE
  mutate(
    ebit_margin = operatingincomeloss / revenue,
    rnd_per_revenue = researchanddevelopmentexpense / revenue,
    capex_margin = capitalexpenditures / revenue,
    capex_per_assets = capitalexpenditures / assets,
    asset_turnover = revenue/ assets,
    intangibleassetsnetincludinggoodwill = goodwill + intangibleassetsnetexcludinggoodwill,
    intangible_intensity = intangibleassetsnetincludinggoodwill / assets,
    revenue_growth = (revenue - lag(revenue)) / lag(revenue),
    current_asset_ratio = assetscurrent / assets,
    
    # ADDING COLUMNS FOR TREATING OUTLIERS - DO NOT TOUCH
    size_log = log(assets)
  )


# Checking for duplicates:
duplicates <- df_cleaning %>%
  mutate(year = year(ymd(end))) %>%
  count(ticker, year) %>%          
  filter(n > 1)                     # Keep only those with more than 1 row

# View the results
print(duplicates)

# Treat the duplicates
df_cleaning <- df_cleaning %>%
  # Group by Ticker and Year to isolate the duplicates
  group_by(ticker, year) %>%
  
  # Keep the report with the latest 'end' date
  slice_max(order_by = end, n = 1) %>%
  
  # Ungroup to return to a normal dataframe
  ungroup()

# Verification: This should now return 0 rows
df_cleaning %>% count(ticker, year) %>% filter(n > 1)

# OUTLIER-check (sorting the outliers by EBIT-margin using IQR-method)
outlier_list <- df_cleaning %>%
  mutate(
    Q1 = quantile(ebit_margin, 0.25),
    Q3 = quantile(ebit_margin, 0.75),
    IQR = Q3 - Q1,
    lower = Q1 - 1.5 * IQR,
    upper = Q3 + 1.5 * IQR
  ) %>%
  filter(ebit_margin < lower | ebit_margin > upper) %>%
  select(company_name, end, ebit_margin)

# Check if there are outliers
print(outlier_list)

# Treating the outliers with Winsorizing, margins set at 1% and 99%.
df_cleaned <- df_cleaning %>%
  mutate(
    ebit_margin_w = Winsorize(ebit_margin, probs = c(0.01, 0.99), na.rm = TRUE),
    asset_turnover_w = Winsorize(asset_turnover, probs = c(0.01, 0.99), na.rm = TRUE),
    rnd_per_revenue_w = Winsorize(rnd_per_revenue, probs = c(0.01, 0.99), na.rm = TRUE),
    capex_margin_w = Winsorize(capex_margin, probs = c(0.01, 0.99), na.rm = TRUE),
    capex_per_assets_w = Winsorize(capex_per_assets, probs = c(0.01, 0.99), na.rm = TRUE),
    intangible_intensity_w = Winsorize(intangible_intensity, probs = c(0.01, 0.99), na.rm = TRUE),
    revenue_growth_w = Winsorize(revenue_growth, probs = c(0.01, 0.99), na.rm = TRUE),
    current_asset_ratio_w = Winsorize(current_asset_ratio, probs = c(0.01, 0.99), na.rm = TRUE)
  )

# Check the results by comparing the original and Winsorized value
summary(df_cleaned$ebit_margin)
summary(df_cleaned$ebit_margin_w)



# Inspect the clean dataset
summary(df_cleaned)
glimpse(df_cleaned)


# Create the Target Variable ("Overperformer")
# We group by YEAR so we compare companies only against peers in the same year.
df_cleaned <- df_cleaned %>%
  group_by(year) %>%
  mutate(
    # Calculate the 80th percentile (top quintile) for this specific year
    threshold_80th = quantile(ebit_margin_w, probs = 0.80, na.rm = TRUE),
    
    # If margin is above threshold, they are an overperformer (1), else (0)
    is_overperformer = if_else(ebit_margin_w >= threshold_80th, 1, 0)
  ) %>%
  ungroup()

# Check the result
table(df_clean$is_overperformer)


# FINAL DATASET TO BE USED
df_final <- df_cleaned %>% select(
  -ebit_margin,
  -asset_turnover,
  -rnd_per_revenue,
  -capex_margin,
  -capex_per_assets,
  -intangible_intensity,
  -revenue_growth,
  -current_asset_ratio
)



# Exporting the data into .csv
write_csv(df_final, "semiconductor_companies_10k_cleaned.csv")
