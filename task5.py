import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.io import write_html

np.random.seed(42)

data = {
    "date": pd.date_range(start="2025-01-01", periods=90, freq="D"),
    "discount": np.random.uniform(5, 30, 90),
    "product_category": np.random.choice(
        ["Electronics", "Clothing", "Food"], 90
    ),
    "region": np.random.choice(["North", "South", "East"], 90),
}

df = pd.DataFrame(data)

daily_avg_discount = df.groupby("date", as_index=False)["discount"].mean()

fig_line = px.line(daily_avg_discount, x="date", y="discount",
                   title="Average Discount Trend")

fig_area = px.area(daily_avg_discount, x="date", y="discount",
                   title="Area Chart")

category_trend = df.groupby(
    ["date", "product_category"], as_index=False
)["discount"].mean()

fig_multi = px.line(category_trend, x="date", y="discount",
                    color="product_category",
                    title="Category Trend")

df["month"] = df["date"].dt.to_period("M").astype(str)

monthly_stats = df.groupby("month")["discount"].agg(
    ["mean", "min", "max", "std"]
).reset_index()

fig_combo = go.Figure()
fig_combo.add_bar(x=monthly_stats["month"],
                  y=monthly_stats["std"],
                  name="Variability")

fig_combo.add_scatter(x=monthly_stats["month"],
                      y=monthly_stats["mean"],
                      mode="lines+markers",
                      name="Average")

fig_combo.update_layout(title="Monthly Stats")

# Simple trend without scipy
trend = np.polyfit(range(len(daily_avg_discount)),
                   daily_avg_discount["discount"], 1)

trend_line = trend[0] * np.arange(len(daily_avg_discount)) + trend[1]

fig_trend = go.Figure()
fig_trend.add_scatter(x=daily_avg_discount["date"],
                      y=daily_avg_discount["discount"],
                      mode="lines",
                      name="Actual")

fig_trend.add_scatter(x=daily_avg_discount["date"],
                      y=trend_line,
                      mode="lines",
                      name="Trend")

# AUTO OPEN
write_html(fig_line, "graph1.html", auto_open=True)
write_html(fig_area, "graph2.html", auto_open=True)
write_html(fig_multi, "graph3.html", auto_open=True)
write_html(fig_combo, "graph4.html", auto_open=True)
write_html(fig_trend, "graph5.html", auto_open=True)

print("✅ 5 Graphs opened!")