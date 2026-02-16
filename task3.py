import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import pearsonr, spearmanr

np.random.seed(0)

# --- Data ---
df = pd.DataFrame({
    "subcategory": np.random.choice([f"subcat_{i}" for i in range(1, 21)], 1000),
    "price": np.random.lognormal(3.0, 0.8, 1000),
    "rating": np.clip(np.random.normal(4.0, 0.6, 1000), 1, 5)
})

grp = (df.groupby("subcategory", as_index=False)
         .agg(avg_price=("price", "mean"),
              avg_rating=("rating", "mean"),
              count=("rating", "size"),
              sd_rating=("rating", "std"))
         .sort_values("avg_price", ascending=False))

# --- Bubble scatter + regression ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=grp, x="avg_price", y="avg_rating",
                size="count", hue="count", sizes=(40, 400),
                palette="viridis", edgecolor="black", alpha=0.85)

sns.regplot(x=np.log(grp["avg_price"]), y=grp["avg_rating"],
            scatter=False, color="crimson")

plt.xscale("log")
plt.xlabel("Avg Price (log)")
plt.ylabel("Avg Rating")
plt.title("Avg Price vs Rating by Subcategory")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --- Correlations ---
pr, pp = pearsonr(np.log(grp["avg_price"]), grp["avg_rating"])
sr, sp = spearmanr(grp["avg_price"], grp["avg_rating"])
print(f"Pearson (log): {pr:.3f}, p={pp:.3g}")
print(f"Spearman: {sr:.3f}, p={sp:.3g}")

# --- Bar + line with error bars ---
x = np.arange(len(grp))
se = grp["sd_rating"] / np.sqrt(grp["count"])

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(x, grp["avg_price"], alpha=0.7)
ax1.set_ylabel("Avg Price")

ax2 = ax1.twinx()
ax2.errorbar(x, grp["avg_rating"], yerr=se, fmt='-o', color="tab:orange")
ax2.set_ylabel("Avg Rating")

ax1.set_xticks(x)
ax1.set_xticklabels(grp["subcategory"], rotation=45, ha="right", fontsize=9)
plt.title("Avg Price & Rating by Subcategory")
plt.tight_layout()
plt.show()

# --- Weighted Pearson ---
x = np.log(grp["avg_price"].values)
y = grp["avg_rating"].values
w = grp["count"].values

wp = np.cov(x, y, aweights=w)[0, 1] / np.sqrt(
     np.cov(x, x, aweights=w)[0, 0] * np.cov(y, y, aweights=w)[0, 0])

print(f"Weighted Pearson (log-price): {wp:.3f}")