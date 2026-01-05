import pandas as pd
import numpy as np

# Life expectancy data (Combined)
LIFE_EXPECTANCY = {
    1900: 47.3,
    1920: 54.1,
    1940: 62.9,
    1950: 68.2,
    1960: 69.7,
    1970: 70.8,
    1980: 73.7,
    1990: 75.4,
    2000: 76.8,
    2010: 78.7,
    2018: 78.7,
    2022: 77.5,
    2026: 77.5  # Assume same as 2022 for predictions
}

def get_life_expectancy(year):
    """Interpolate life expectancy for a given year."""
    years = sorted(LIFE_EXPECTANCY.keys())

    # If exact match, return it
    if year in LIFE_EXPECTANCY:
        return LIFE_EXPECTANCY[year]

    # Find surrounding years for interpolation
    lower_year = max([y for y in years if y <= year], default=years[0])
    upper_year = min([y for y in years if y >= year], default=years[-1])

    if lower_year == upper_year:
        return LIFE_EXPECTANCY[lower_year]

    # Linear interpolation
    lower_le = LIFE_EXPECTANCY[lower_year]
    upper_le = LIFE_EXPECTANCY[upper_year]
    ratio = (year - lower_year) / (upper_year - lower_year)
    return lower_le + ratio * (upper_le - lower_le)

# Load the expanded 1900-2025 dataset
df = pd.read_csv("../SCOTUS_Justices_All_Years_1900-2025.xlsx - Justices by Year.csv")

print(f"Loaded dataset: {len(df)} records")
print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")

print(f"\nOriginal Status distribution:")
print(df['Status'].value_counts())

# Calculate Age-to-Life-Expectancy Ratio
df['Life Expectancy'] = df['Year'].apply(get_life_expectancy)
df['Age Ratio'] = df['Age'] / df['Life Expectancy']

print(f"\nAge Ratio statistics:")
print(f"  Min: {df['Age Ratio'].min():.3f}")
print(f"  Max: {df['Age Ratio'].max():.3f}")
print(f"  Mean: {df['Age Ratio'].mean():.3f}")

# Create Ideology Alignment feature
def get_ideology_alignment(row):
    president_party = row["President Party"]
    ideology = row["Ideology"]

    if ideology == "Moderate":
        return "Neutral"
    elif (president_party == "Democratic" and ideology == "Conservative") or \
         (president_party == "Republican" and ideology == "Liberal"):
        return "Opposing"
    else:
        return "Aligned"

df["Ideology Alignment"] = df.apply(get_ideology_alignment, axis=1)

# Drop columns not needed for training (including original Age, keeping Age Ratio)
columns_to_drop = [
    'Year',
    'Justice Name',
    'Appointing President',
    'Year Appointed',
    'Age',  # Using Age Ratio instead
    'Life Expectancy',
    'President',
    'Departure Reason'  # Not needed for prediction
]
df = df.drop(columns=columns_to_drop)

# Convert Status to binary (0 = Stayed, 1 = Left - any type)
# This includes both voluntary AND involuntary departures
df['Status'] = df['Status'].apply(lambda x: 0 if x == 'Stayed' else 1)

print(f"\nBinary Status distribution (0=Stayed, 1=Left):")
print(df['Status'].value_counts())
print(f"Departure rate: {df['Status'].mean()*100:.2f}%")

# Columns to convert to dummy variables
categorical_columns = [
    "Appointing President Party",
    "President Party",
    "Ideology",
    "House Control",
    "Senate Control",
    "Election Year Type",
    "Ideology Alignment"
]

# Create dummy variables (drop_first=True to avoid multicollinearity)
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Save the cleaned data
df.to_csv("cleaned_training_data.csv", index=False)

print("\nData cleaned and saved to cleaned_training_data.csv")
print(f"\nShape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nFirst few rows:\n{df.head()}")

# Show election year type columns
election_cols = [c for c in df.columns if 'Election Year Type' in c]
print(f"\nElection Year Type columns: {election_cols}")

# Show ideology alignment columns
alignment_cols = [c for c in df.columns if 'Ideology Alignment' in c]
print(f"Ideology Alignment columns: {alignment_cols}")
