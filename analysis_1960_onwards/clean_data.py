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

# Load the data
df = pd.read_csv("../SCOTUS_Justices_by_Election_Year.xlsx - Justices by Election Year.csv")

# Filter for 1960 onwards
df = df[df['Election Year'] >= 1960]
print(f"Filtered to 1960 onwards: {len(df)} records")
print(f"Year range: {df['Election Year'].min()} - {df['Election Year'].max()}")

# Calculate Age-to-Life-Expectancy Ratio
df['Life Expectancy'] = df['Election Year'].apply(get_life_expectancy)
df['Age Ratio'] = df['Age in Election Year'] / df['Life Expectancy']

print(f"\nAge Ratio statistics:")
print(f"  Min: {df['Age Ratio'].min():.3f}")
print(f"  Max: {df['Age Ratio'].max():.3f}")
print(f"  Mean: {df['Age Ratio'].mean():.3f}")

# Create Ideology Alignment feature before dropping columns
def get_ideology_alignment(row):
    president_party = row["President's Party"]
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
columns_to_drop = ['Election Year', 'Justice Name', 'President', 'Appointed By', 'Year Appointed', 'Age in Election Year', 'Life Expectancy']
df = df.drop(columns=columns_to_drop)

# Convert Status to binary (0 = Stayed, 1 = Left)
df['Status'] = df['Status'].apply(lambda x: 0 if x == 'Stayed' else 1)

# Columns to convert to dummy variables
categorical_columns = [
    "Appointing President's Party",
    "President's Party",
    "Ideology",
    "House Control",
    "Senate Control",
    "President's Party Controls",
    "Election Type",
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
print(f"\nStatus distribution:\n{df['Status'].value_counts()}")

# Show ideology alignment columns
alignment_cols = [c for c in df.columns if 'Ideology Alignment' in c]
print(f"\nIdeology Alignment columns: {alignment_cols}")
