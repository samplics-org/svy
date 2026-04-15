import polars as pl
import svy


df = pl.DataFrame(
    {
        "region": ["North", "South", "North", "East", "South"],
        "age": [25, 45, 30, 55, 35],
        "income": [30000, 50000, 40000, 60000, 45000],
    }
)
s = svy.Sample(df)

print(s.data)


# Filter with dict (equality/membership)
s_north = s.wrangling.filter_records({"region": "North"})
print(s_north.data)

s_north_south = s.wrangling.filter_records({"region": ["North", "South"]})
print(s_north_south.data)
