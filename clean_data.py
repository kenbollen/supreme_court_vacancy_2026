import pandas as pd

# Load the data
df = pd.read_csv("SCOTUS_Justices_by_Election_Year.xlsx - Justices by Election Year.csv")

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

# Drop columns not needed for training
columns_to_drop = ['Election Year', 'Justice Name', 'President', 'Appointed By', 'Year Appointed']
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

print("Data cleaned and saved to cleaned_training_data.csv")
print(f"\nShape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nStatus distribution:\n{df['Status'].value_counts()}")

# Show ideology alignment columns
alignment_cols = [c for c in df.columns if 'Ideology Alignment' in c]
print(f"\nIdeology Alignment columns: {alignment_cols}")
