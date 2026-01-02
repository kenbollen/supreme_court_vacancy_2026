import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 60)
print("HISTORICAL VISUALIZATIONS - 1960 ONWARDS")
print("=" * 60)

# Load the raw data and filter for 1960+
df = pd.read_csv("../SCOTUS_Justices_by_Election_Year.xlsx - Justices by Election Year.csv")
df = df[df['Election Year'] >= 1960]
print(f"\nFiltered to 1960 onwards: {len(df)} records")
print(f"Year range: {df['Election Year'].min()} - {df['Election Year'].max()}")

# Create Ideology Alignment feature
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

# Create departure type column
def get_departure_type(status):
    if status == "Stayed":
        return "Stayed"
    elif "Voluntary" in status or "Retired" in status:
        return "Voluntary"
    else:
        return "Involuntary"

df["Departure Type"] = df["Status"].apply(get_departure_type)

# Filter to only departures
departures = df[df["Status"] != "Stayed"].copy()

print(f"Total departures (1960+): {len(departures)}")
print(f"\nDeparture types:\n{departures['Departure Type'].value_counts()}")

# Create figure with subplots
fig = plt.figure(figsize=(16, 14))
fig.suptitle('Supreme Court Justice Departures Analysis (1960 Onwards)', fontsize=16, fontweight='bold', y=1.02)

# ============================================================
# PLOT 1: Departures by Ideology Alignment
# ============================================================
ax1 = fig.add_subplot(2, 2, 1)

alignment_counts = departures["Ideology Alignment"].value_counts()
colors = {'Aligned': '#2ecc71', 'Opposing': '#e74c3c', 'Neutral': '#f39c12'}
bar_colors = [colors.get(x, '#95a5a6') for x in alignment_counts.index]

bars = ax1.bar(alignment_counts.index, alignment_counts.values, color=bar_colors, edgecolor='black')

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

ax1.set_xlabel('Ideology Alignment with President', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Departures', fontsize=12, fontweight='bold')
ax1.set_title('Justice Departures by Ideology Alignment\n(1960-2024)', fontsize=14, fontweight='bold')

# ============================================================
# PLOT 2: Age at Departure Over Time (Voluntary vs Involuntary)
# ============================================================
ax2 = fig.add_subplot(2, 2, 2)

voluntary = departures[departures["Departure Type"] == "Voluntary"]
involuntary = departures[departures["Departure Type"] == "Involuntary"]

ax2.scatter(voluntary["Election Year"], voluntary["Age in Election Year"],
            c='#3498db', label='Voluntary', alpha=0.7, s=80, edgecolor='black')
ax2.scatter(involuntary["Election Year"], involuntary["Age in Election Year"],
            c='#e74c3c', label='Involuntary', alpha=0.7, s=80, marker='X', edgecolor='black')

# Add trend lines if enough data
if len(voluntary) > 1:
    z_vol = np.polyfit(voluntary["Election Year"], voluntary["Age in Election Year"], 1)
    p_vol = np.poly1d(z_vol)
    x_range = np.linspace(voluntary["Election Year"].min(), voluntary["Election Year"].max(), 100)
    ax2.plot(x_range, p_vol(x_range),
             '--', color='#3498db', alpha=0.8, linewidth=2, label='Voluntary Trend')

if len(involuntary) > 1:
    z_inv = np.polyfit(involuntary["Election Year"], involuntary["Age in Election Year"], 1)
    p_inv = np.poly1d(z_inv)
    x_range = np.linspace(involuntary["Election Year"].min(), involuntary["Election Year"].max(), 100)
    ax2.plot(x_range, p_inv(x_range),
             '--', color='#e74c3c', alpha=0.8, linewidth=2, label='Involuntary Trend')

ax2.set_xlabel('Election Year', fontsize=12, fontweight='bold')
ax2.set_ylabel('Age at Departure', fontsize=12, fontweight='bold')
ax2.set_title('Age at Departure Over Time\nVoluntary vs Involuntary', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left')

# ============================================================
# PLOT 3: Departures by Congress Control (Opposing Ideology)
# ============================================================
ax3 = fig.add_subplot(2, 2, 3)

opposing_departures = departures[departures["Ideology Alignment"] == "Opposing"].copy()

def get_opposing_congress_control(row):
    justice_ideology = row["Ideology"]
    house = row["House Control"]
    senate = row["Senate Control"]

    if justice_ideology == "Liberal":
        opposing_party = "Republican"
    elif justice_ideology == "Conservative":
        opposing_party = "Democratic"
    else:
        return "N/A"

    house_opposing = house == opposing_party
    senate_opposing = senate == opposing_party

    if house_opposing and senate_opposing:
        return "Both Chambers"
    elif not house_opposing and not senate_opposing:
        return "Neither Chamber"
    else:
        return "One Chamber"

if len(opposing_departures) > 0:
    opposing_departures["Opposing Congress Control"] = opposing_departures.apply(get_opposing_congress_control, axis=1)
    opp_congress_counts = opposing_departures["Opposing Congress Control"].value_counts()
    order = ["Both Chambers", "One Chamber", "Neither Chamber"]
    opp_congress_counts = opp_congress_counts.reindex([x for x in order if x in opp_congress_counts.index])

    colors_congress = {'Both Chambers': '#c0392b', 'One Chamber': '#e67e22', 'Neither Chamber': '#27ae60'}
    bar_colors_3 = [colors_congress.get(x, '#95a5a6') for x in opp_congress_counts.index]

    bars3 = ax3.bar(opp_congress_counts.index, opp_congress_counts.values, color=bar_colors_3, edgecolor='black')

    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=14, fontweight='bold')
else:
    ax3.text(0.5, 0.5, 'No opposing ideology\ndepartures in this period',
             ha='center', va='center', fontsize=12, transform=ax3.transAxes)

ax3.set_xlabel('Opposing Party Congress Control', fontsize=12, fontweight='bold')
ax3.set_ylabel('Number of Departures', fontsize=12, fontweight='bold')
ax3.set_title('Departures When Justice Has OPPOSING Ideology\nBy Opposing Party Congress Control', fontsize=14, fontweight='bold')

# ============================================================
# PLOT 4: Departures by Congress Control (Aligned Ideology)
# ============================================================
ax4 = fig.add_subplot(2, 2, 4)

aligned_departures = departures[departures["Ideology Alignment"] == "Aligned"].copy()

def get_aligned_congress_control(row):
    justice_ideology = row["Ideology"]
    house = row["House Control"]
    senate = row["Senate Control"]

    if justice_ideology == "Liberal":
        aligned_party = "Democratic"
    elif justice_ideology == "Conservative":
        aligned_party = "Republican"
    else:
        return "N/A"

    house_aligned = house == aligned_party
    senate_aligned = senate == aligned_party

    if house_aligned and senate_aligned:
        return "Both Chambers"
    elif not house_aligned and not senate_aligned:
        return "Neither Chamber"
    else:
        return "One Chamber"

if len(aligned_departures) > 0:
    aligned_departures["Aligned Congress Control"] = aligned_departures.apply(get_aligned_congress_control, axis=1)
    aligned_congress_counts = aligned_departures["Aligned Congress Control"].value_counts()
    aligned_congress_counts = aligned_congress_counts.reindex([x for x in order if x in aligned_congress_counts.index])

    colors_congress_aligned = {'Both Chambers': '#27ae60', 'One Chamber': '#f39c12', 'Neither Chamber': '#c0392b'}
    bar_colors_4 = [colors_congress_aligned.get(x, '#95a5a6') for x in aligned_congress_counts.index]

    bars4 = ax4.bar(aligned_congress_counts.index, aligned_congress_counts.values, color=bar_colors_4, edgecolor='black')

    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=14, fontweight='bold')
else:
    ax4.text(0.5, 0.5, 'No aligned ideology\ndepartures in this period',
             ha='center', va='center', fontsize=12, transform=ax4.transAxes)

ax4.set_xlabel('Aligned Party Congress Control', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Departures', fontsize=12, fontweight='bold')
ax4.set_title('Departures When Justice Has ALIGNED Ideology\nBy Aligned Party Congress Control', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('historical_departures_analysis_1960_onwards.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("SUMMARY STATISTICS (1960 ONWARDS)")
print("=" * 60)

print("\n1. Departures by Ideology Alignment:")
print(departures["Ideology Alignment"].value_counts().to_string())

print("\n2. Age at Departure Statistics:")
if len(voluntary) > 0:
    print(f"   Voluntary - Mean: {voluntary['Age in Election Year'].mean():.1f}, Median: {voluntary['Age in Election Year'].median():.1f}")
if len(involuntary) > 0:
    print(f"   Involuntary - Mean: {involuntary['Age in Election Year'].mean():.1f}, Median: {involuntary['Age in Election Year'].median():.1f}")

print("\n3. Opposing Ideology Departures by Congress Control:")
if len(opposing_departures) > 0:
    print(opposing_departures["Opposing Congress Control"].value_counts().to_string())
else:
    print("   No opposing ideology departures")

print("\n4. Aligned Ideology Departures by Congress Control:")
if len(aligned_departures) > 0:
    print(aligned_departures["Aligned Congress Control"].value_counts().to_string())
else:
    print("   No aligned ideology departures")

print("\n\nChart saved to: historical_departures_analysis_1960_onwards.png")
