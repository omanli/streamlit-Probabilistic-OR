"""
Arrival Process Dynamics
Emergence of Exponential Interarrival Times from Renewing Events
"""

import streamlit as st
import altair as alt
import numpy as np
import pandas as pd

st.set_page_config(page_title="Arrival Events")
st.markdown("# Arrival Events")
st.sidebar.header("Arrival Events")
st.write("This demo tool simulates arrival events from multiple renewal processes.")
st.write("Enjoy!")
st.write("")


distr_par = {
    None : '--',
    'Exponential' : [1, 'Exponential(avg)'], 
    'Uniform'     : [2, 'Uniform(a,b)'], 
    'Triangular'  : [3, 'Triangular(a,m,b)'],
}


@st.cache_data
def generate_event_times(rs, D, par, N, n):
    if D is None:
        return None, None
    if D not in distr_par:
        raise ValueError(f"Incorrect distribution selection={D!r}")

    rng = np.random.RandomState(rs)
    par = par.replace(',', ' ')
    param = [float(p) for p in par.split()]
    if len(param) != distr_par[distribution][0]:
        raise ValueError(f"Incorrect parameter values={param!r} for distribution")

    I = getattr(rng, D.lower())(*param, size=(N,n))
    
    A = np.cumsum(I, axis=1)

    E = sorted([(A[i,k], src[i]) for i in range(N) for k in range(n)])

    T = float(round(E[-1][0], 1))

    return E, T




cols = st.columns([1,1,1])
with cols[0]:
    random_seed = st.number_input("random seed:", value=123)
    distribution = st.selectbox(
        label='Interarrival Time Distribution:',
        options=('Exponential', 'Uniform', 'Triangular'),
        index=None,
        placeholder='Choose a Distribution')
with cols[1]:
    num_sources  = st.number_input("number of sources:", value=3)
    param_str = st.text_input(distr_par[distribution][1], value='')
with cols[2]:
    num_events  = st.number_input("number of arrivals:", value=10)
with cols[0]:
    st.write(f'You selected: {distribution}({param_str})')


src  = [f"A{i+1:02}" for i in range(num_sources)]


try:
    E,T = generate_event_times(
            random_seed,
            distribution,
            param_str,
            num_sources,
            num_events)
except ValueError as e:
    # e.message
    st.error(getattr(e, 'message', repr(e)))


t_range = st.slider(label='Plot Time Window', 
                    min_value=0,
                    max_value=int(T),
                    step=1,
                    value=(int(0.3*T), int(0.7*T)))


data = []
for i in range(num_sources + 1):
    if i == 0:
        d = dict(t=[e[0] for e in E], e=[e[1] for e in E])
    else:
        d = dict(t=[e[0] for e in E if e[1]==src[i-1]], e=["A" for e in E if e[1]==src[i-1]])
    data.append(pd.DataFrame(d))
    idx = (data[-1].t >= t_range[0]) & (data[-1].t <= t_range[1])
    data[-1] = data[-1].loc[idx]


charts = []
charts.append(alt.Chart(data[0])
                 .mark_point()
                 .encode(x=alt.X("t:Q", scale=alt.Scale(domain=[t_range[0],t_range[1]])),
                         y=alt.X("e:N")))
for i in range(num_sources+1):
    charts.append(alt.Chart(data[i])
                 .mark_point()
                 .encode(x=alt.X("t:Q", scale=alt.Scale(domain=[t_range[0],t_range[1]])),
                         y=alt.X("e:N")))
LC = alt.layer(*charts)

st.altair_chart(LC, use_container_width=True)

# st.dataframe(data[0])
# st.write(dict(t=[e[0] for e in E if e[1]==src[-1]], e=["A" for e in E if e[1]==src[-1]]))

st.button("Re-run")
