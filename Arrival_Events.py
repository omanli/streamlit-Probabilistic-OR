"""
Arrival Process Dynamics
Emergence of Exponential Interarrival Times from Renewing Events
"""

import streamlit as st
import altair as alt
import numpy as np
import pandas as pd

st.set_page_config(page_title="Arrival Events")

def dmy():
    rng = np.random.RandomState(rs)
    A = np.cumsum(rng.uniform(0, 1, size=(N,n)), axis=1)
    E = np.random.multivariate_normal(mean=[0, 0], cov=[[3, 3], [3, 4]], size=500)
    X = rng.multivariate_normal(mean=[0, 0], cov=[[3, 3], [3, 4]], size=500)
    plt.stairs

    for i in range(1, 101):
        new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        chart.add_rows(new_rows)
        last_rows = new_rows


st.markdown("# Arrival Events")
st.sidebar.header("Arrival Events")
st.write("This demo tool simulates arrival events.")
st.write("Enjoy!")
st.write("")

rs = 123

N = 3
n = 10
Rt = 1/4

src  = [f"A{i+1:02}" for i in range(N)]

A = np.cumsum(np.random.exponential(1/Rt, size=(N,n)), axis=1)
E = [(A[i,k], src[i]) for i in range(N) for k in range(n)]
E.sort()

data = []
data.append(pd.DataFrame(dict(t=[e[0] for e in E], e=[e[1] for e in E])))
for i in range(N):
    d = dict(t=[e[0] for e in E if e[1]==src[i]], e=["A" for e in E if e[1]==src[i]])
    data.append(pd.DataFrame(d))

charts = []
charts.append(alt.Chart(data[0])
                 .mark_point()
                 .encode(x="t:Q", y="e:N"))
for i in range(N+1):
    charts.append(alt.Chart(data[i])
                 .mark_point()
                 .encode(x="t:Q", y="e:N"))
LC = alt.layer(*charts)

st.altair_chart(LC, use_container_width=True)

st.dataframe(data[0])

st.button("Re-run")
