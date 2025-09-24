import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx

st.set_page_config(page_title="Datacolor - PSM Addition related Analysis", layout="wide")
st.title("Datacolor - PSM Addition related Analysis")

# ---------- CONFIG ----------
SHADE_COL = "SHADE No."
QUALITY_COL = "QUALITY"
MACHINE_COL = "M/C"
BATCH_COL = "BATCH No."
SM_USED_COL = "SM add used by GL?"
HAD_TO_USE_COL = "HAD TO USE?"
PSM_POINTS = "No. of PSM Points Available"

# Extra columns for hover
RECIPE_COL = "RECIPE DYESTUFF ORDER"
TOTAL_DYE_COL = "TOTAL DYE DEPTH (%)"
LR_COL = "L/R"
GREIGE_WEIGHT_COL = "Greige Weight (kg)"
PROCESS_COL = "Combined Process (Chemical Profile)"
PRIMARY_DE_COL = "Primary dE Value"
SOP_COL = "Standard SOP? (Not altered by Wicky)"
DYES_COUNT_COL = "No.of Main Dyes"

# ---------- File Upload ----------
uploaded = st.file_uploader(
    "Upload Excel (must contain 'Ex - Main Recipe Analysis' & 'Ex - SM Add Analysis')",
    type=["xlsx", "xls"]
)
if uploaded is None:
    st.info("Upload the Excel file to begin.")
    st.stop()

xls = pd.ExcelFile(uploaded)
required_sheets = {"Ex - Main Recipe Analysis", "Ex - SM Add Analysis"}
if not required_sheets.issubset(set(xls.sheet_names)):
    st.error(f"Excel must contain sheets: {required_sheets}. Found: {xls.sheet_names}")
    st.stop()

main_df = pd.read_excel(xls, sheet_name="Ex - Main Recipe Analysis")
sm_df = pd.read_excel(xls, sheet_name="Ex - SM Add Analysis")

if BATCH_COL not in main_df.columns or BATCH_COL not in sm_df.columns:
    st.error(f"Both sheets must contain column '{BATCH_COL}' to merge.")
    st.stop()

sm_df = sm_df.drop_duplicates(subset=[BATCH_COL], keep="first")
df = pd.merge(
    main_df,
    sm_df[[BATCH_COL, SM_USED_COL, HAD_TO_USE_COL, PSM_POINTS]],
    on=BATCH_COL,
    how="left"
).drop_duplicates(subset=[BATCH_COL], keep="first").reset_index(drop=True)

data = df.copy()

# ---------- Normalize ----------
if SHADE_COL in data.columns:
    data[SHADE_COL] = (
        data[SHADE_COL].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    )
for c in [QUALITY_COL, MACHINE_COL, SM_USED_COL, HAD_TO_USE_COL]:
    if c in data.columns:
        data[c] = data[c].astype(str).str.strip().str.upper()

# ---------- Tree Function ----------
def draw_tree(df, mode="Shade", root_name="Root"):
    tree_df = df.copy()
    tree_df["SM_group"] = tree_df[SM_USED_COL].str.upper().replace(
        {"APPROXIMATELY USED": "APPROXIMATELY USED", "YES": "Yes", "NO": "No"}
    )
    tree_df.loc[~tree_df["SM_group"].isin(["APPROXIMATELY USED", "Yes", "No"]), "SM_group"] = tree_df[SM_USED_COL]

    total_count = len(tree_df)
    G = nx.DiGraph()
    G.add_node("Root", level=0, node_type="root",
               label=f"{root_name}\n({total_count})",
               count=total_count,
               color="#FF6B6B")

    # fixed colors for SM usage options
    sm_color_map = {
        "Yes": "#e74c3c",  # red
        "NO": "#3498db",   # blue
        "APPROXIMATELY USED": "#f39c12"
    }

    # outcome colors
    outcome_values = sorted(tree_df[HAD_TO_USE_COL].dropna().unique().tolist())
    palette = [
        "#2ecc71", "#9b59b6", "#16a085", "#e67e22", "#8e44ad", "#1abc9c",
        "#2c3e50", "#d35400", "#7f8c8d", "#c0392b", "#27ae60", "#34495e"
    ]
    outcome_color_map = {val: palette[i % len(palette)] for i, val in enumerate(outcome_values)}

    outcome_hover = {}

    if mode == "Shade":
        grouped = tree_df.groupby([QUALITY_COL, MACHINE_COL, "SM_group", HAD_TO_USE_COL]).size().reset_index(name="count")
        quality_totals = tree_df.groupby(QUALITY_COL).size().reset_index(name="quality_total")
        machine_totals = tree_df.groupby([QUALITY_COL, MACHINE_COL]).size().reset_index(name="machine_total")
        usage_totals = tree_df.groupby([QUALITY_COL, MACHINE_COL, "SM_group"]).size().reset_index(name="usage_total")

        for _, row in grouped.iterrows():
            quality = row[QUALITY_COL]; machine = row[MACHINE_COL]
            sm_group = row["SM_group"]; outcome = row[HAD_TO_USE_COL]; count = row["count"]

            quality_total = quality_totals.loc[quality_totals[QUALITY_COL] == quality, "quality_total"].values[0]
            machine_total = machine_totals.loc[
                (machine_totals[QUALITY_COL] == quality) & (machine_totals[MACHINE_COL] == machine),
                "machine_total"
            ].values[0]
            usage_total = usage_totals.loc[
                (usage_totals[QUALITY_COL] == quality) &
                (usage_totals[MACHINE_COL] == machine) &
                (usage_totals["SM_group"] == sm_group),
                "usage_total"
            ].values[0]

            quality_label = f"{quality}\n({quality_total})"
            machine_label = f"{machine}\n({machine_total})"
            usage_label = f"{sm_group}\n({usage_total})"
            outcome_label = f"{outcome}\n({count})"

            outcome_node = f"outcome_{quality}_{machine}_{sm_group}_{outcome}"

            subset = tree_df[
                (tree_df[QUALITY_COL]==quality) &
                (tree_df[MACHINE_COL]==machine) &
                (tree_df["SM_group"]==sm_group) &
                (tree_df[HAD_TO_USE_COL]==outcome)
            ]
            hover_texts=[]
            for _, r in subset.iterrows():
                hover_texts.append(
                    f"<b>Batch No:</b> {r.get(BATCH_COL,'')}<br>"
                    f"<b>RECIPE DYESTUFF ORDER:</b> {r.get(RECIPE_COL,'')}<br>"
                    f"<b>TOTAL DYE DEPTH (%):</b> {r.get(TOTAL_DYE_COL,'')}<br>"
                    f"<b>L/R:</b> {r.get(LR_COL,'')}<br>"
                    f"<b>Combined Process (Chemical Profile):</b> {r.get(PROCESS_COL,'')}<br>"
                    f"<b>Greige Weight (kg):</b> {r.get(GREIGE_WEIGHT_COL,'')}<br>"
                    f"<b>Primary dE Value:</b> {r.get(PRIMARY_DE_COL,'')}<br>"
                    f"<b>Standard SOP? (Not altered by Wicky):</b> {r.get(SOP_COL,'')}<br>"
                    f"<b>No.of Main Dyes:</b> {r.get(DYES_COUNT_COL,'')}<br>"
                    f"<b>No. of PSM Points Available:</b> {r.get(PSM_POINTS,'')}<br>"
                )
            outcome_hover[outcome_node]="<br><br>".join(hover_texts)

            # add nodes
            G.add_node(f"quality_{quality}", level=1, node_type="quality", label=quality_label, count=quality_total, color="#FFA07A")
            G.add_node(f"machine_{quality}_{machine}", level=2, node_type="machine", label=machine_label, count=machine_total, color="#4ECDC4")
            usage_color = sm_color_map.get(sm_group.upper(), "#45B7D1")
            G.add_node(f"usage_{quality}_{machine}_{sm_group}", level=3, node_type="usage", label=usage_label, count=usage_total, color=usage_color)
            outcome_color = outcome_color_map.get(outcome, "#96CEB4")
            G.add_node(outcome_node, level=4, node_type="outcome", label=outcome_label, count=count, color=outcome_color)

            G.add_edge("Root", f"quality_{quality}")
            G.add_edge(f"quality_{quality}", f"machine_{quality}_{machine}")
            G.add_edge(f"machine_{quality}_{machine}", f"usage_{quality}_{machine}_{sm_group}")
            G.add_edge(f"usage_{quality}_{machine}_{sm_group}", outcome_node)

    elif mode == "Machine":
        shade_totals = tree_df.groupby(SHADE_COL).size().reset_index(name="shade_total")
        quality_totals_global = tree_df.groupby(QUALITY_COL).size().reset_index(name="quality_total")

        grouped = tree_df.groupby([SHADE_COL, QUALITY_COL, "SM_group", HAD_TO_USE_COL]).size().reset_index(name="count")
        usage_totals = tree_df.groupby([SHADE_COL, QUALITY_COL, "SM_group"]).size().reset_index(name="usage_total")
        shade_quality_totals = tree_df.groupby([SHADE_COL, QUALITY_COL]).size().reset_index(name="shade_quality_total")

        for _, row in grouped.iterrows():
            shade = row[SHADE_COL]; quality = row[QUALITY_COL]
            sm_group = row["SM_group"]; outcome = row[HAD_TO_USE_COL]; count = row["count"]

            shade_total = shade_totals.loc[shade_totals[SHADE_COL] == shade, "shade_total"].values[0]
            shade_quality_total = shade_quality_totals.loc[
                (shade_quality_totals[SHADE_COL] == shade) & (shade_quality_totals[QUALITY_COL] == quality),
                "shade_quality_total"
            ].values[0]
            usage_total = usage_totals.loc[
                (usage_totals[SHADE_COL] == shade) &
                (usage_totals[QUALITY_COL] == quality) &
                (usage_totals["SM_group"] == sm_group),
                "usage_total"
            ].values[0]

            shade_label = f"{shade}\n({shade_total})"
            quality_label = f"{quality}\n({shade_quality_total})"
            usage_label = f"{sm_group}\n({usage_total})"
            outcome_label = f"{outcome}\n({count})"

            outcome_node = f"outcome_{shade}_{quality}_{sm_group}_{outcome}"

            subset = tree_df[
                (tree_df[SHADE_COL]==shade) &
                (tree_df[QUALITY_COL]==quality) &
                (tree_df["SM_group"]==sm_group) &
                (tree_df[HAD_TO_USE_COL]==outcome)
            ]
            hover_texts=[]
            for _, r in subset.iterrows():
                hover_texts.append(
                    f"<b>Batch No:</b> {r.get(BATCH_COL,'')}<br>"
                    f"<b>RECIPE DYESTUFF ORDER:</b> {r.get(RECIPE_COL,'')}<br>"
                    f"<b>TOTAL DYE DEPTH (%):</b> {r.get(TOTAL_DYE_COL,'')}<br>"
                    f"<b>L/R:</b> {r.get(LR_COL,'')}<br>"
                    f"<b>Combined Process (Chemical Profile):</b> {r.get(PROCESS_COL,'')}<br>"
                    f"<b>Greige Weight (kg):</b> {r.get(GREIGE_WEIGHT_COL,'')}<br>"
                    f"<b>Primary dE Value:</b> {r.get(PRIMARY_DE_COL,'')}<br>"
                    f"<b>Standard SOP? (Not altered by Wicky):</b> {r.get(SOP_COL,'')}<br>"
                    f"<b>No.of Main Dyes:</b> {r.get(DYES_COUNT_COL,'')}<br>"
                    f"<b>No. of PSM Points Available:</b> {r.get(PSM_POINTS,'')}<br>"
                )
            outcome_hover[outcome_node] = "<br><br>".join(hover_texts)

            # add nodes
            G.add_node(f"shade_{shade}", level=1, node_type="shade", label=shade_label, count=shade_total, color="#FFD166")
            G.add_node(f"quality_{shade}_{quality}", level=2, node_type="quality", label=quality_label, count=shade_quality_total, color="#FFA07A")
            usage_color = sm_color_map.get(sm_group.upper(), "#45B7D1")
            G.add_node(f"usage_{shade}_{quality}_{sm_group}", level=3, node_type="usage", label=usage_label, count=usage_total, color=usage_color)
            outcome_color = outcome_color_map.get(outcome, "#96CEB4")
            G.add_node(outcome_node, level=4, node_type="outcome", label=outcome_label, count=count, color=outcome_color)

            # add edges
            if not G.has_edge("Root", f"shade_{shade}"):
                G.add_edge("Root", f"shade_{shade}")
            if not G.has_edge(f"shade_{shade}", f"quality_{shade}_{quality}"):
                G.add_edge(f"shade_{shade}", f"quality_{shade}_{quality}")
            if not G.has_edge(f"quality_{shade}_{quality}", f"usage_{shade}_{quality}_{sm_group}"):
                G.add_edge(f"quality_{shade}_{quality}", f"usage_{shade}_{quality}_{sm_group}")
            G.add_edge(f"usage_{shade}_{quality}_{sm_group}", outcome_node)

    # --- Layout ---
    pos, levels = {}, {}
    for node, d in G.nodes(data=True):
        levels.setdefault(d["level"], []).append(node)

    for level, nodes in levels.items():
        spacing = max(3.5, len(nodes) * 1.2)
        for i, node in enumerate(nodes):
            pos[node] = (level,(i - len(nodes)/2) * spacing)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1,color="#ccc"), hoverinfo="none")

    traces=[]
    node_types = ["root", "shade", "quality", "machine", "usage", "outcome"]
    for nt in node_types:
        xs, ys, labels, sizes, hovers, colors_list = [], [], [], [], [], []
        for node, d in G.nodes(data=True):
            if d["node_type"] == nt:
                xs.append(pos[node][0]); ys.append(pos[node][1])
                labels.append(d["label"])
                sizes.append(30 + len(str(d["label"])) * 0.6)
                if nt == "outcome":
                    hovers.append(outcome_hover.get(node, d.get("label", "")))
                else:
                    hovers.append(d.get("label", ""))
                colors_list.append(d.get("color", "#96CEB4"))
        if xs:
            traces.append(go.Scatter(
                x=xs, y=ys, mode="markers+text", text=labels, textposition="middle center",
                marker=dict(size=sizes, color=colors_list, line=dict(width=2, color="white")),
                name=nt.capitalize(), hoverinfo="text", hovertext=hovers
            ))

    fig = go.Figure(data=[edge_trace] + traces)
    fig.update_layout(
        title="Tree Plot",
        showlegend=True, hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=650, width=None, autosize=True, plot_bgcolor="white",
        hoverlabel=dict(align="left")

    )


    if mode == "Shade":
        st.markdown("**Hierarchy Path:** Shade No → Quality → Machine → SM add used by GL? → HAD TO USE?")
        st.markdown(f"**Selected Shade:** {root_name}")
    else:
        st.markdown("**Hierarchy Path:** Machine → Shade → Quality → SM add used by GL? → HAD TO USE?")
        st.markdown(f"**Selected Machine:** {root_name}")

    st.plotly_chart(fig, use_container_width=True)

# ---------- Sidebar Mode ----------
st.sidebar.header("Select Mode")
mode = st.sidebar.radio("Analyze by", ["Shade", "Machine"], index=0)

# ---------- Shade ----------
if mode == "Shade":
    all_shades = sorted(data[SHADE_COL].dropna().unique().tolist())
    selected_shade = st.sidebar.selectbox("Shade", all_shades,index=5)
    qualities = sorted(data.loc[data[SHADE_COL] == selected_shade, QUALITY_COL].dropna().unique().tolist())
    selected_qualities = st.sidebar.multiselect("Quality", options=qualities, default=qualities)

    filtered = data[(data[SHADE_COL] == selected_shade) & (data[QUALITY_COL].isin(selected_qualities))]
    if filtered.empty:
        st.warning("No data")
        st.stop()

    st.header(f"Aggregated results for Shade {selected_shade}")
    total = len(filtered)
    gl_used = (filtered[SM_USED_COL].isin(["YES", "APPROXIMATELY USED"])).sum()
    num_machines = filtered[MACHINE_COL].nunique()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Batches", total)
    col2.metric("GL Used SM Adds (%)", f"{gl_used/total*100:.1f}%")
    col3.metric("GL Not Used SM Adds (%)", f"{(total-gl_used)/total*100:.1f}%")
    col4.metric("Machines", num_machines)

    draw_tree(filtered, mode="Shade", root_name=f"{selected_shade}")

# ---------- Machine ----------
elif mode == "Machine":
    all_machines = sorted(data[MACHINE_COL].dropna().unique().tolist())
    selected_machine = st.sidebar.selectbox("Machine", all_machines)
    qualities = sorted(data.loc[data[MACHINE_COL] == selected_machine, QUALITY_COL].dropna().unique().tolist())
    selected_qualities = st.sidebar.multiselect("Quality", options=qualities, default=qualities)

    filtered = data[(data[MACHINE_COL] == selected_machine) & (data[QUALITY_COL].isin(selected_qualities))]
    if filtered.empty:
        st.warning("No data")
        st.stop()

    st.header(f"Aggregated results for Machine {selected_machine}")
    total = len(filtered)
    gl_used = (filtered[SM_USED_COL].isin(["YES", "APPROXIMATELY USED"])).sum()

    uniq_shades = filtered[SHADE_COL].dropna().unique().tolist()
    uniq_qualities = filtered[QUALITY_COL].dropna().unique().tolist()
    num_shades = len(uniq_shades)
    num_qualities = len(uniq_qualities)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Batches", total)
    col2.metric("GL Used SM Adds (%)", f"{gl_used/total*100:.1f}%")
    col3.metric("GL Not Used SM Adds (%)", f"{(total-gl_used)/total*100:.1f}%")
    col4.metric("Shades", f"{num_shades}")
    col5.metric("Qualities", f"{num_qualities}")

    draw_tree(filtered, mode="Machine", root_name=f"{selected_machine}")


import plotly.graph_objects as go

# ---------- Convert numeric columns ----------
for col in [LR_COL, TOTAL_DYE_COL, PRIMARY_DE_COL]:
    filtered[col] = pd.to_numeric(filtered[col], errors='coerce')

# ---------- Determine x-axis ----------
if mode == "Machine":
    x_col = SHADE_COL  # x-axis = shade numbers
    x_title = "Shade No"
elif mode == "Shade":
    x_col = MACHINE_COL  # x-axis = machine numbers
    x_title = "Machine"

# Convert x-axis to string (to treat as categorical, exact numbers)
filtered['x_str'] = filtered[x_col].astype(str)

# ---------- L/R Distribution ----------
fig_lr = go.Figure()
fig_lr.add_trace(go.Scatter(
    x=filtered['x_str'],
    y=filtered[LR_COL],
    mode='markers',
    marker=dict(size=8, color='#1f77b4'),
    line=dict(color='#1f77b4', width=1),
    text=[f"Batch: {b}<br>L/R: {lr}<br>{x_col}: {xv}" 
          for b, lr, xv in zip(filtered[BATCH_COL], filtered[LR_COL], filtered['x_str'])],
    hoverinfo='text'
))
fig_lr.update_layout(
    title="L/R Distribution",
    xaxis_title=x_title,
    yaxis_title="L/R Value",
    xaxis=dict(type='category', tickangle=-45),  # categorical x-axis
    height=450,
    plot_bgcolor="white",
    hovermode="closest"
)
fig_lr.add_hline(y=10, line_dash="dash", line_color="red",
                 annotation_text="⬆️ Normal", annotation_position="top left", line_width=1)
fig_lr.add_hline(y=20, line_dash="dash", line_color="red",
                 annotation_text="⬆️ High L/R", annotation_position="top left", line_width=1)
st.plotly_chart(fig_lr, use_container_width=True)

# ---------- Total Dye Depth (%) ----------
fig_td = go.Figure()
fig_td.add_trace(go.Scatter(
    x=filtered['x_str'],
    y=filtered[TOTAL_DYE_COL],
    mode='markers',
    marker=dict(size=8, color='#1f77b4'),
    line=dict(color='#1f77b4', width=1),
    text=[f"Batch: {b}<br>Total Dye Depth: {td}<br>{x_col}: {xv}" 
          for b, td, xv in zip(filtered[BATCH_COL], filtered[TOTAL_DYE_COL], filtered['x_str'])],
    hoverinfo='text'
))
fig_td.update_layout(
    title="Total Dye Depth (%) Distribution",
    xaxis_title=x_title,
    yaxis_title="Total Dye Depth (%)",
    xaxis=dict(type='category', tickangle=-45),
    height=450,
    plot_bgcolor="white",
    hovermode="closest"
)
st.plotly_chart(fig_td, use_container_width=True)

# ---------- Primary dE Value ----------
fig_de = go.Figure()
fig_de.add_trace(go.Scatter(
    x=filtered['x_str'],
    y=filtered[PRIMARY_DE_COL],
    mode='markers',
    marker=dict(size=8, color='#1f77b4'),
    line=dict(color='#1f77b4', width=1),
    text=[f"Batch: {b}<br>Primary dE: {de}<br>{x_col}: {xv}" 
          for b, de, xv in zip(filtered[BATCH_COL], filtered[PRIMARY_DE_COL], filtered['x_str'])],
    hoverinfo='text'
))
fig_de.update_layout(
    title="Primary dE Value Distribution",
    xaxis_title=x_title,
    yaxis_title="Primary dE",
    xaxis=dict(type='category', tickangle=-45),
    height=450,
    plot_bgcolor="white",
    hovermode="closest"
)
fig_de.add_hline(y=0.9, line_dash="dash", line_color="red",
                  annotation_text="⬇️ Low dE", annotation_position="top left", line_width=1)
fig_de.add_hline(y=3, line_dash="dash", line_color="red",
                  annotation_text="⬆️ Shade out", annotation_position="top left", line_width=1)
st.plotly_chart(fig_de, use_container_width=True)
