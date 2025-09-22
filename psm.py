import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from st_aggrid import AgGrid, GridOptionsBuilder  # For interactive tables

st.set_page_config(page_title="Datacolor - PSM Addition related analysis", layout="wide")
st.title("Datacolor - PSM Addition related analysis")

# ---------- CONFIG: exact expected column names ----------
SHADE_COL = "SHADE No."
QUALITY_COL = "QUALITY"
MACHINE_COL = "M/C"
BATCH_COL = "BATCH No."
SM_USED_COL = "SM add used by GL?"
HAD_TO_USE_COL = "HAD TO USE?"
PSM_POINTS = "No. of PSM Points Available"

# Additional columns for detailed table
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

# ---------- Load & merge ----------
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
)

df = df.drop_duplicates(subset=[BATCH_COL], keep="first").reset_index(drop=True)
data = df.copy()

# ---------- Column Normalization ----------
if SHADE_COL in data.columns:
    data[SHADE_COL] = (
        data[SHADE_COL]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

for c in [QUALITY_COL, MACHINE_COL, SM_USED_COL, HAD_TO_USE_COL]:
    if c in data.columns:
        data[c] = data[c].astype(str).str.strip().str.upper()

# ---------- Sidebar filters ----------
st.sidebar.header("Select Shade & Quality")
all_shades = data[SHADE_COL].dropna().unique().tolist()


def shade_sort_key(shade):
    try:
        return (0, float(shade)) if shade.replace(".", "").isdigit() else (1, shade)
    except:
        return (1, str(shade))


all_shades_sorted = sorted(all_shades, key=shade_sort_key)

if all_shades_sorted:
    selected_shade = st.sidebar.selectbox("Shade", all_shades_sorted, index=5)
else:
    selected_shade = None
    st.sidebar.warning("No shades found in data")

if selected_shade:
    qualities = sorted(
        data.loc[data[SHADE_COL] == selected_shade, QUALITY_COL].dropna().unique().tolist()
    )
else:
    qualities = sorted(data[QUALITY_COL].dropna().unique().tolist())

selected_qualities = st.sidebar.multiselect(
    "Quality (choose one or more)", options=qualities, default=qualities
)

# ---------- Helpers ----------
def compute_machine_summary(df_subset):
    rows = []
    machines = df_subset[MACHINE_COL].dropna().unique().tolist()
    for m in sorted(machines):
        g = df_subset[df_subset[MACHINE_COL] == m]
        total = len(g)
        if total == 0:
            continue
        gl_used_count = len(
            g[g[SM_USED_COL].isin(["YES", "USED", "APPROXIMATELY USED"])]
        )
        gl_not_used_count = total - gl_used_count
        rows.append(
            {
                "Machine": m,
                "Total Runs": total,
                "GL Used (count)": gl_used_count,
                "GL Used (%)": round(100 * gl_used_count / total, 1),
                "GL Not Used (count)": gl_not_used_count,
                "GL Not Used (%)": round(100 * gl_not_used_count / total, 1),
            }
        )
    return pd.DataFrame(rows)


def show_machine_metrics_and_charts(df_subset, title=""):
    st.markdown(f"## {title}" if title else "### Summary")
    machine_summary = compute_machine_summary(df_subset)
    if machine_summary.empty:
        st.warning("No machine data to show here.")
        return

    # KPIs
    total_runs = int(machine_summary["Total Runs"].sum())
    total_gl_used = int(machine_summary["GL Used (count)"].sum())
    overall_gl_used_pct = round(100 * total_gl_used / total_runs, 1) if total_runs > 0 else 0
    overall_gl_not_used_pct = (
        round(100 * (1 - (total_gl_used / total_runs)), 1) if total_runs > 0 else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Runs", total_runs)
    col2.metric("GL Used SM Adds (overall %)", f"{overall_gl_used_pct}%")
    col3.metric("GL Not Used SM Adds (overall %)", f"{overall_gl_not_used_pct}%")
    col4.metric("Machines", machine_summary.shape[0])

    # ---------- Tree plot ----------
    st.subheader("Quality-wise SM Point Usage vs Outcome")

    tree_df = df_subset.copy()
    tree_df["SM_group"] = tree_df[SM_USED_COL].str.upper().replace(
        {"APPROXIMATELY USED": "APPROXIMATELY USED", "YES": "Yes", "NO": "No"}
    )
    tree_df.loc[~tree_df["SM_group"].isin(["APPROXIMATELY USED", "Yes","No"]), "SM_group"] = tree_df[SM_USED_COL]

    total_count = len(tree_df)
    G = nx.DiGraph()
    G.add_node(
        "Root",
        level=0,
        node_type="root",
        label=f"All Batches\n({total_count}, 100%)",
        count=total_count,
    )

    grouped = (
        tree_df.groupby([QUALITY_COL, MACHINE_COL, "SM_group", HAD_TO_USE_COL])
        .size()
        .reset_index(name="count")
    )

    quality_totals = tree_df.groupby(QUALITY_COL).size().reset_index(name="quality_total")
    machine_totals = tree_df.groupby([QUALITY_COL, MACHINE_COL]).size().reset_index(name="machine_total")
    usage_totals = tree_df.groupby([QUALITY_COL, MACHINE_COL, "SM_group"]).size().reset_index(name="usage_total")

    outcome_details = {}

    for _, row in grouped.iterrows():
        quality = row[QUALITY_COL]
        machine = row[MACHINE_COL]
        sm_group = row["SM_group"]
        outcome = row[HAD_TO_USE_COL]
        count = row["count"]

        quality_total = quality_totals.loc[quality_totals[QUALITY_COL] == quality, "quality_total"].values[0]
        machine_total = machine_totals.loc[
            (machine_totals[QUALITY_COL] == quality) & (machine_totals[MACHINE_COL] == machine), "machine_total"
        ].values[0]
        usage_total = usage_totals.loc[
            (usage_totals[QUALITY_COL] == quality)
            & (usage_totals[MACHINE_COL] == machine)
            & (usage_totals["SM_group"] == sm_group), "usage_total"
        ].values[0]

        quality_label = f"{quality}\n({quality_total}, {round((quality_total/total_count)*100, 1)}%)"
        machine_label = f"{machine}\n({machine_total}, {round((machine_total/quality_total)*100, 1)}%)"
        usage_label = f"{sm_group}\n({usage_total}, {round((usage_total/machine_total)*100, 1)}%)"
        outcome_label = f"{outcome}\n({count}, {round((count/usage_total)*100, 1)}%)"

        G.add_node(f"quality_{quality}", level=1, node_type="quality", label=quality_label, count=quality_total)
        G.add_node(f"machine_{quality}_{machine}", level=2, node_type="machine", label=machine_label, count=machine_total)
        G.add_node(f"usage_{quality}_{machine}_{sm_group}", level=3, node_type="usage", label=usage_label, count=usage_total)

        outcome_node_id = f"outcome_{quality}_{machine}_{sm_group}_{outcome}"

        # --- outcome hover details ---
        outcome_rows = tree_df[
            (tree_df[QUALITY_COL] == quality)
            & (tree_df[MACHINE_COL] == machine)
            & (tree_df["SM_group"] == sm_group)
            & (tree_df[HAD_TO_USE_COL] == outcome)
        ]

        details_list = []
        for _, r in outcome_rows.iterrows():
            details = (
                f"Batch No: {r[BATCH_COL]}<br>"
                f"RECIPE DYESTUFF ORDER: {r[RECIPE_COL]}<br>"
                f"TOTAL DYE DEPTH (%): {r[TOTAL_DYE_COL]}<br>"
                f"L/R: {r[LR_COL]}<br>"
                f"Greige Weight (kg): {r[GREIGE_WEIGHT_COL]}<br>"
                f"Combined Process (Chemical Profile): {r[PROCESS_COL]}<br>"
                f"Primary dE Value: {r[PRIMARY_DE_COL]}<br>"
                f"PSM Points: {r[PSM_POINTS]}<br>"
                f"Standard SOP? (Not altered by Wicky): {r[SOP_COL]}<br>"
                f"No.of Main Dyes: {r[DYES_COUNT_COL]}"
            )
            details_list.append(details)

        hover_details = "<br>".join(details_list)
        outcome_details[outcome_node_id] = hover_details

        G.add_node(outcome_node_id, level=4, node_type="outcome", label=outcome_label, count=count, details=hover_details)

        G.add_edge("Root", f"quality_{quality}")
        G.add_edge(f"quality_{quality}", f"machine_{quality}_{machine}")
        G.add_edge(f"machine_{quality}_{machine}", f"usage_{quality}_{machine}_{sm_group}")
        G.add_edge(f"usage_{quality}_{machine}_{sm_group}", outcome_node_id)

    # Layout
    pos, levels = {}, {}
    for node, data in G.nodes(data=True):
        levels.setdefault(data["level"], []).append(node)

    for level, nodes in levels.items():
        for i, node in enumerate(nodes):
            horizontal_spacing = 3.5
            x_pos = (i - len(nodes) / 2) * horizontal_spacing
            pos[node] = (x_pos, -level)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=2, color="#888"), hoverinfo="none", mode="lines")

    node_traces = {}
    colors = {"root": "#FF6B6B", "quality": "#FFA07A", "machine": "#4ECDC4", "usage": "#45B7D1", "outcome": "#96CEB4"}

    def calculate_node_size(label, node_type, count):
        base_sizes = {"root": 60, "quality": 55, "machine": 50, "usage": 45, "outcome": 40}
        text_length = len(label)
        size_adjustment = text_length * 0.7
        count_adjustment = count * 0.08 if count > 0 else 0
        return base_sizes[node_type] + size_adjustment + count_adjustment

    for node_type in ["root", "quality", "machine", "usage", "outcome"]:
        node_x, node_y, node_labels, node_sizes, node_info = [], [], [], [], []
        for node, data in G.nodes(data=True):
            if data["node_type"] == node_type:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_labels.append(data["label"])
                node_sizes.append(calculate_node_size(data["label"], node_type, data.get("count", 0)))

                info = f"{data['label']}<br>"
                if node_type == "outcome":
                    info += data.get("details", "")
                elif "count" in data:
                    info += f"Absolute count: {data['count']}<br>"
                node_info.append(info)

        if node_x:
            node_traces[node_type] = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=node_labels,
                textposition="middle center",
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=node_info,
                marker=dict(size=node_sizes, color=colors[node_type], line=dict(width=2, color="white")),
                name=node_type.capitalize(),
                textfont=dict(size=10, color="black", family="Arial Black, sans-serif"),
                hoverlabel=dict(font_size=14 if node_type=="outcome" else 12)  # <-- outcome hover bigger
            )

    fig_tree = go.Figure(data=[edge_trace] + list(node_traces.values()))
    fig_tree.update_layout(
        title=dict(
            text="Tree Plot: Quality → Machine → SM Add Used by GL? → HAD TO USE?",
            font=dict(size=16),
        ),
        showlegend=True,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=80),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        height=700,
        width=1200,
    )
    st.plotly_chart(fig_tree, use_container_width=True)


# ---------- Apply filters and display ----------
filtered = data.copy()
if selected_shade:
    filtered = filtered[filtered[SHADE_COL] == selected_shade]
if selected_qualities:
    filtered = filtered[filtered[QUALITY_COL].isin(selected_qualities)]

if filtered.empty:
    st.warning("No data for the selected filters.")
    st.stop()

st.header("Aggregated results for selected Shade & Quality")
show_machine_metrics_and_charts(
    filtered, title=f"{selected_shade if selected_shade else 'All Shades'}"
)
