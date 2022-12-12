import plotly.express as px
import pandas as pd
import geopandas as gpd
import numpy as np
import json


df = pd.read_csv("../parsed.csv")
session_stat = df.groupby("session").size()
session_stat.head()
session_stat[session_stat > 100]

gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.lat, df.lon), crs="EPSG:4326"
)

tester_session = "928675d7-29b7-410a-a7ae-629a0126352e"
gdf_tester = gdf[gdf.session == tester_session]
# print(gdf_tester)


fig = px.scatter_mapbox(
    gdf_tester,
    lat="lat",
    lon="lon",
    mapbox_style="carto-positron",
    zoom=16,
    size="size",
    size_max=10,
    # animation_frame="datetime",
    animation_frame="seq_NO",
)

# fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 10
# fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 5
fig.update_geos(projection_type="equirectangular", visible=True, resolution=50)
fig.show()
