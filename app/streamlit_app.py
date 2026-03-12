import streamlit as st

from topocompass import SpinExchangeModel, solve_magnon_bilinears


st.set_page_config(page_title="TopoCompass", layout="wide")
st.title("TopoCompass")
st.caption(
    "Automated generator and solver of magnon bilinears for C3i-symmetric honeycomb Mott insulators"
)

st.subheader("Model Inputs")
col1, col2, col3 = st.columns(3)
with col1:
    j = st.number_input("J", value=1.0)
with col2:
    k = st.number_input("K", value=0.0)
with col3:
    gamma = st.number_input("Gamma", value=0.0)

hx = st.slider("Hx", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
hy = st.slider("Hy", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
hz = st.slider("Hz", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

if st.button("Generate & Solve"):
    model = SpinExchangeModel(
        exchanges={"J": j, "K": k, "Gamma": gamma},
        magnetic_field_xyz=(hx, hy, hz),
        symmetry="C3i",
    )
    result = solve_magnon_bilinears(model)
    st.success("Computation finished")
    st.json(result)
