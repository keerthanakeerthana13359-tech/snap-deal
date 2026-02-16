import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# -----------------------------
# Data simulation (weak correlation)
# -----------------------------
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "Discount": np.random.uniform(0, 50, n),
})
df["Rating"] = np.clip(3 + 0.01 * df["Discount"] + np.random.normal(0, 1, n), 1, 5)

# -----------------------------
# Correlation
# -----------------------------
corr, p = pearsonr(df["Discount"], df["Rating"])
print(f"Correlation: {corr:.3f}, P-value: {p:.3f}")

# -----------------------------
# Scatter plot
# -----------------------------
plt.scatter(df["Discount"], df["Rating"], alpha=0.4)
plt.xlabel("Discount (%)")
plt.ylabel("Rating")
plt.title("Rating vs Discount")
plt.grid(alpha=0.3)
plt.show()

# -----------------------------
# Discount bins + average rating
# -----------------------------
df["Discount_Bin"] = pd.cut(
    df["Discount"],
    [0, 10, 20, 30, 40, 50],
    labels=["0–10%", "10–20%", "20–30%", "30–40%", "40–50%"]
)

avg_rating = df.groupby("Discount_Bin", observed=True)["Rating"].mean().reset_index()

sns.barplot(x="Discount_Bin", y="Rating", data=avg_rating)
plt.ylim(1, 5)
plt.title("Average Rating by Discount Range")
plt.show()

# -----------------------------
# Counterexamples
# -----------------------------
print("\nHigh discount & low rating:")
print(df[(df.Discount > 40) & (df.Rating < 2.5)].head(3)[["Discount", "Rating"]])

print("\nLow discount & high rating:")
print(df[(df.Discount < 10) & (df.Rating > 4)].head(3)[["Discount", "Rating"]])

# -----------------------------
# Conclusion
# -----------------------------
print("\nConclusion:",
      "Weak relationship between discount and rating"
      if abs(corr) < 0.3 else "Noticeable correlation exists")