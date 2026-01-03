import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

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
    if year in LIFE_EXPECTANCY:
        return LIFE_EXPECTANCY[year]
    lower_year = max([y for y in years if y <= year], default=years[0])
    upper_year = min([y for y in years if y >= year], default=years[-1])
    if lower_year == upper_year:
        return LIFE_EXPECTANCY[lower_year]
    lower_le = LIFE_EXPECTANCY[lower_year]
    upper_le = LIFE_EXPECTANCY[upper_year]
    ratio = (year - lower_year) / (upper_year - lower_year)
    return lower_le + ratio * (upper_le - lower_le)

print("=" * 60)
print("SUPREME COURT VACANCY MODEL - 1960 ONWARDS")
print("(Using Age-to-Life-Expectancy Ratio)")
print("=" * 60)

# Load the cleaned data
df = pd.read_csv("cleaned_training_data.csv")

# Separate features and target
X = df.drop(columns=['Status'])
y = df['Status']

print("\nDataset Overview")
print("=" * 50)
print(f"Total samples: {len(df)}")
print(f"Features: {X.shape[1]}")
print(f"Target distribution: Stayed={sum(y==0)}, Left={sum(y==1)}")
print(f"\nFeatures used:\n{X.columns.tolist()}")

# Split data (stratified to maintain class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Scale features for better convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression with Lasso (L1) regularization
model = LogisticRegression(
    penalty='l1',
    solver='saga',
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
print("\n" + "=" * 50)
print("Model Performance")
print("=" * 50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                 Predicted")
print(f"                 Stayed  Left")
print(f"Actual Stayed    {cm[0][0]:5d}  {cm[0][1]:5d}")
if len(cm) > 1:
    print(f"Actual Left      {cm[1][0]:5d}  {cm[1][1]:5d}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Stayed', 'Left'], zero_division=0))

# Feature importance (coefficients)
print("=" * 50)
print("Feature Coefficients (Impact on Leaving)")
print("=" * 50)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

# Show selected features (non-zero coefficients)
selected = feature_importance[feature_importance['Coefficient'] != 0]
eliminated = feature_importance[feature_importance['Coefficient'] == 0]

print(f"\nSelected Features ({len(selected)}):")
print("-" * 40)
for _, row in selected.iterrows():
    direction = "+" if row['Coefficient'] > 0 else "-"
    print(f"{direction} {row['Feature']}: {row['Coefficient']:.4f}")

print(f"\nEliminated by Lasso ({len(eliminated)}):")
print("-" * 40)
for _, row in eliminated.iterrows():
    print(f"  {row['Feature']}")

# ============================================================
# PREDICT 2026 SUPREME COURT VACANCIES
# ============================================================
print("\n" + "=" * 60)
print("2026 SUPREME COURT VACANCY PREDICTIONS")
print("=" * 60)

# Load raw 2026 prediction data
df_2026_raw = pd.read_csv("../prediction_set_raw.csv")

# Store justice names for output
justice_names = df_2026_raw['Justice Name'].tolist()

# Apply same transformations as training data
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

df_2026_raw["Ideology Alignment"] = df_2026_raw.apply(get_ideology_alignment, axis=1)

# Calculate Age Ratio for 2026 predictions
life_exp_2026 = get_life_expectancy(2026)
df_2026_raw['Age Ratio'] = df_2026_raw['Age in Election Year'] / life_exp_2026

print(f"\n2026 Life Expectancy: {life_exp_2026}")
print("Justice Age Ratios:")
for _, row in df_2026_raw.iterrows():
    print(f"  {row['Justice Name']}: {row['Age in Election Year']} / {life_exp_2026:.1f} = {row['Age Ratio']:.3f}")

# Drop columns not needed (including raw Age, keeping Age Ratio)
columns_to_drop = ['Election Year', 'Justice Name', 'President', 'Appointed By', 'Year Appointed', 'Status', 'Age in Election Year']
df_2026 = df_2026_raw.drop(columns=columns_to_drop)

# Create dummy variables
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
df_2026 = pd.get_dummies(df_2026, columns=categorical_columns, drop_first=True)

# Ensure all columns from training are present (add missing as False)
for col in X.columns:
    if col not in df_2026.columns:
        df_2026[col] = False

# Ensure column order matches training data
df_2026 = df_2026[X.columns]

# Scale features using same scaler
X_2026_scaled = scaler.transform(df_2026)

# Predict probabilities
proba_leave = model.predict_proba(X_2026_scaled)[:, 1]
proba_stay = model.predict_proba(X_2026_scaled)[:, 0]

# Display individual predictions
print("\nIndividual Justice Predictions:")
print("-" * 60)
print(f"{'Justice':<25} {'P(Leave)':<12} {'P(Stay)':<12}")
print("-" * 60)

predictions = []
for name, p_leave, p_stay in zip(justice_names, proba_leave, proba_stay):
    print(f"{name:<25} {p_leave:>8.1%}      {p_stay:>8.1%}")
    predictions.append({'Justice': name, 'P(Leave)': p_leave, 'P(Stay)': p_stay})

# Sort by probability of leaving
print("\nRanked by Likelihood to Leave:")
print("-" * 60)
sorted_predictions = sorted(predictions, key=lambda x: x['P(Leave)'], reverse=True)
for i, pred in enumerate(sorted_predictions, 1):
    print(f"{i}. {pred['Justice']:<25} {pred['P(Leave)']:>8.1%}")

# Calculate probability at least 1 justice leaves
prob_all_stay = 1.0
for p_stay in proba_stay:
    prob_all_stay *= p_stay

prob_at_least_one_leaves = 1 - prob_all_stay

print("\n" + "=" * 60)
print("OVERALL VACANCY PROBABILITY")
print("=" * 60)
print(f"P(All justices stay):        {prob_all_stay:>8.1%}")
print(f"P(At least 1 vacancy):       {prob_at_least_one_leaves:>8.1%}")
print("=" * 60)

# ============================================================
# VISUALIZATION
# ============================================================

# Create bar chart of predictions
fig, ax = plt.subplots(figsize=(12, 7))

# Sort by probability for better visualization
sorted_data = sorted(zip(justice_names, proba_leave), key=lambda x: x[1], reverse=True)
names_sorted = [x[0] for x in sorted_data]
proba_sorted = [x[1] * 100 for x in sorted_data]

# Color bars by ideology alignment
colors = []
for name in names_sorted:
    row = df_2026_raw[df_2026_raw['Justice Name'] == name].iloc[0]
    alignment = row['Ideology Alignment']
    if alignment == 'Aligned':
        colors.append('#2ecc71')
    elif alignment == 'Opposing':
        colors.append('#e74c3c')
    else:
        colors.append('#f39c12')

bars = ax.barh(names_sorted, proba_sorted, color=colors, edgecolor='black', linewidth=0.5)

# Add value labels on bars
for bar, prob in zip(bars, proba_sorted):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{prob:.1f}%', va='center', fontsize=11, fontweight='bold')

# Styling
ax.set_xlabel('Probability of Leaving (%)', fontsize=12, fontweight='bold')
ax.set_title('2026 Supreme Court Vacancy Predictions\n(1960+ Data with Age-to-Life-Expectancy Ratio)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, max(proba_sorted) + 3)
ax.invert_yaxis()

# Add legend
legend_elements = [
    Patch(facecolor='#2ecc71', edgecolor='black', label='Aligned with President'),
    Patch(facecolor='#e74c3c', edgecolor='black', label='Opposing President'),
    Patch(facecolor='#f39c12', edgecolor='black', label='Neutral (Moderate)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Add overall probability annotation
ax.annotate(f'P(At least 1 vacancy) = {prob_at_least_one_leaves:.1%}',
            xy=(0.98, 0.02), xycoords='axes fraction',
            fontsize=12, fontweight='bold',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('2026_vacancy_predictions_1960_onwards.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved to: 2026_vacancy_predictions_1960_onwards.png")
