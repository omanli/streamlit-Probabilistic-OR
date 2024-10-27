"""
Arrival Process Dynamics
Emergence of Exponential Interarrival Times from Renewing Events
"""

import streamlit as st
import altair as alt
import numpy as np
import pandas as pd

st.set_page_config(page_title="Arrival Events", layout="wide")
st.markdown("## Arrival Events")
st.sidebar.header("Arrival Events")
st.write("This demo simulates arrival events from multiple renewal processes. Enjoy!")
st.write("\n")
st.write("\n")


distr_par = {
    None : '--',
    'Exponential' : [1, '(avg)'], 
    'Uniform'     : [2, '(a,b)'], 
    'Triangular'  : [3, '(a,m,b)'],
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
    if len(param) == 0:
        raise ValueError(f"Parameter values empty")
    if len(param) != distr_par[distribution][0]:
        raise ValueError(f"Incorrect parameter values={param!r} for distribution")

    I = getattr(rng, D.lower())(*param, size=(N,n))
    A = np.cumsum(I, axis=1)
    m = A[:,-1].min()

    E = sorted([(A[i,k], src[i]) for i in range(N) for k in range(n) if A[i,k] <= m])
    T = float(round(E[-1][0], 1))

    return E, T



cols_inp1 = st.columns([1,1,1])
with cols_inp1[0]:
    random_seed = st.number_input("random seed:", value=123)
with cols_inp1[1]:
    num_sources = st.number_input("number of sources:", value=3)
with cols_inp1[2]:
    num_events = st.number_input("number of arrivals:", value=10)

cols_inp2 = st.columns([2,2,1,1])
with cols_inp2[0]:
    distribution = st.selectbox(
        label='Interarrival Time Distribution:',
        options=tuple(d for d in distr_par.keys() if d),
        index=None,
        placeholder='Choose a Distribution')
with cols_inp2[1]:
    param_str = st.text_input(f"parameters {distr_par[distribution][1]}:", value='')
with cols_inp2[2]:
    t_beg = st.number_input("t_begin:", value=0)
with cols_inp2[3]:
    t_end = st.number_input("t_end:", value=1)




src  = [f"A{i+1:02}" for i in range(num_sources)]

try:
    E,T = generate_event_times(random_seed,
                               distribution,
                               param_str,
                               num_sources,
                               num_events)
except ValueError as e:
    st.error(getattr(e, 'message', repr(e)))
    E,T = None, None


if E:
    t_range = st.slider(label='Time Window:', 
                        min_value=0,
                        max_value=int(np.ceil(T)),
                        step=1,
                        value=(t_beg, t_end))

    st.write(f"{t_range}\n")

    data = []
    for i in range(num_sources + 1):
        if i == 0:
            d = dict(t=[e[0] for e in E], e=[e[1] for e in E])
        else:
            d = dict(t=[e[0] for e in E if e[1]==src[i-1]], e=["A" for e in E if e[1]==src[i-1]])
        data.append(pd.DataFrame(d))
        idx = (data[-1].t >= t_range[0]) & (data[-1].t <= t_range[1])
        data[-1] = data[-1].loc[idx]

    chart_e = []
    chart_e.append(alt.Chart(data[0])
                      .mark_point()
                      .encode(x=alt.X("t:Q", 
                                    scale=alt.Scale(domain=t_range)),
                              y=alt.X("e:N")))
    for i in range(num_sources+1):
        chart_e.append(alt.Chart(data[i])
                          .mark_point()
                          .encode(x=alt.X("t:Q", 
                                          scale=alt.Scale(domain=t_range)),
                                  y=alt.X("e:N")))

    idx = (data[0].t >= t_beg) & (data[0].t <= t_end)
    data[0]['dt'] = data[0].loc[idx].t.diff()
    chart_h = (alt.Chart(data[0].loc[idx])
                  .mark_bar()
                  .encode(x=alt.X('dt:Q', title="interarrival times",
                          bin=alt.BinParams(maxbins=min(30,2*np.ceil(np.sqrt(num_sources*num_events))))),
                          y=alt.Y('count()', title="frequency")))

    cols_charts = st.columns([3,1])
    with cols_charts[0]:
        st.altair_chart(alt.layer(*chart_e), 
                        use_container_width=True)
    with cols_charts[1]:
        st.altair_chart(chart_h,
                        use_container_width=True)

    # how to dump data on screen:
    # st.dataframe(data[0])
    # st.write(dict(t=[e[0] for e in E if e[1]==src[-1]], e=["A" for e in E if e[1]==src[-1]]))

st.button("Re-run")
