import base64
from io import BytesIO
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any, Dict, List

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.optimize as spo
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import requests

DATA_ENV_PATH = "HOP_DATA_PATH"
DATA_ENV_SHAREPOINT_URL = "SHAREPOINT_FILE_URL"
DATA_ENV_SHAREPOINT_TOKEN = "SHAREPOINT_ACCESS_TOKEN"

MIN_HOP_RATE_PER_HOP = 0.15
MAX_HOP_RATE_PER_HOP = 5
MAX_HOPS_IN_BLEND = 10
MAX_COMPOUNDS_TO_PRIORITIZE = 10


class DataLoadError(Exception):
    pass


def _fetch_sharepoint_bytes(sharepoint_url: str, token: str | None) -> bytes:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.get(sharepoint_url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.content


def _load_excel_bytes(source_path: str | None, sharepoint_url: str | None, token: str | None) -> bytes:
    if sharepoint_url:
        return _fetch_sharepoint_bytes(sharepoint_url, token)
    if not source_path:
        source_path = os.getenv(DATA_ENV_PATH, "Hop_Blend_Model_Built v3.xlsx")
    if not os.path.exists(source_path):
        raise DataLoadError(f"Excel file not found: {source_path}")
    with open(source_path, "rb") as handle:
        return handle.read()


def clean_compound_name(name: Any) -> Any:
    if isinstance(name, str):
        name = name.replace("\u2009", " ").replace("\xa0", " ").strip()
        name = " ".join(name.split())
    return name


def build_state(data_bytes: bytes) -> Dict[str, Any]:
    df_hop_results = pd.read_excel(BytesIO(data_bytes), sheet_name="Hop_Results")
    df_replace_hops = pd.read_excel(BytesIO(data_bytes), sheet_name="Replace_Hops")

    df_hop_results["Compound"] = df_hop_results["Compound"].apply(clean_compound_name)
    df_replace_hops["Compound"] = df_replace_hops["Compound"].apply(clean_compound_name)

    unique_compounds_hop_results = df_hop_results["Compound"].unique()
    unique_compounds_replace_hops = df_replace_hops["Compound"].unique()
    all_unique_compounds = pd.Series(
        list(unique_compounds_hop_results) + list(unique_compounds_replace_hops)
    ).unique()

    compound_to_class_map: Dict[str, str] = {}
    for compound in all_unique_compounds:
        compound_lower = compound.lower()
        if "ester" in compound_lower:
            compound_to_class_map[compound] = "Esters"
        elif (
            "terpene" in compound_lower
            or "myrcene" in compound_lower
            or "linalool" in compound_lower
            or "geraniol" in compound_lower
            or "pinene" in compound_lower
            or "humulene" in compound_lower
            or "farnesene" in compound_lower
        ):
            compound_to_class_map[compound] = "Terpenes"
        elif "alcohol" in compound_lower or (
            "ol" in compound_lower
            and "linalool" not in compound_lower
            and "geraniol" not in compound_lower
        ):
            compound_to_class_map[compound] = "Alcohols"
        elif "ketone" in compound_lower or "one" in compound_lower:
            compound_to_class_map[compound] = "Ketones"
        elif "aldehyde" in compound_lower or "al" in compound_lower:
            compound_to_class_map[compound] = "Aldehydes"
        elif "acid" in compound_lower:
            compound_to_class_map[compound] = "Acids"
        else:
            compound_to_class_map[compound] = "Miscellaneous"

    unique_chemical_classes = sorted(list(set(compound_to_class_map.values())))

    max_concentration_factor = max(
        df_hop_results["Concentration"].max(),
        df_replace_hops["Concentration"].max(),
    )

    hop_profiles_for_clustering: Dict[str, Dict[str, float]] = {}

    def populate_hop_profiles(df: pd.DataFrame) -> None:
        for hop_name in df["Hop"].unique():
            hop_data = df[df["Hop"] == hop_name]
            profile = {
                row["Compound"]: row["Concentration"]
                for _, row in hop_data.iterrows()
            }
            hop_profiles_for_clustering[hop_name] = profile

    populate_hop_profiles(df_hop_results)
    populate_hop_profiles(df_replace_hops)

    df_hop_features = pd.DataFrame.from_dict(
        hop_profiles_for_clustering, orient="index"
    )
    df_hop_features = df_hop_features.reindex(columns=all_unique_compounds).fillna(0)

    scaler = StandardScaler()
    df_hop_features_scaled = scaler.fit_transform(df_hop_features)
    df_hop_features_scaled = pd.DataFrame(
        df_hop_features_scaled,
        columns=df_hop_features.columns,
        index=df_hop_features.index,
    )

    linked_matrix = linkage(df_hop_features_scaled, method="ward", metric="euclidean")
    num_clusters = 5
    clusters = fcluster(linked_matrix, num_clusters, criterion="maxclust")

    df_hop_features["Cluster"] = clusters
    hop_to_cluster_map = df_hop_features["Cluster"].to_dict()

    compound_oav_map = {
        "Linalool": 1000,
        "Geraniol": 800,
        "beta-Pinene": 600,
        "Myrcene": 500,
        "Humulene": 400,
        "alpha-Pinene": 350,
        "2-Methylbutyl isobutyrate": 950,
        "2-Nonanone": 750,
        "Methyl geranate": 900,
        "limonene": 250,
        "beta-Caryophyllene": 300,
        "alpha-Farnesene": 200,
        "delta-Cadinene": 150,
        "Methyl hexanoate": 700,
        "Ethyl isobutyrate": 650,
        "Ethyl 2-methylbutanoate": 550,
        "Isobutyl acetate": 450,
        "1-Decanol": 50,
        "1-Hexanol": 30,
        "2-Decanol": 20,
        "2-Heptanone": 100,
        "2-Dodecanone": 80,
        "Citronellol": 120,
        "alpha-Terpineol": 110,
        "Guaiacol": 5000,
        "Hexyl acetate": 200,
        "Isoamyl acetate": 180,
    }

    return {
        "df_hop_results": df_hop_results,
        "df_replace_hops": df_replace_hops,
        "all_unique_compounds": all_unique_compounds,
        "compound_to_class_map": compound_to_class_map,
        "unique_chemical_classes": unique_chemical_classes,
        "max_concentration_factor": max_concentration_factor,
        "df_hop_features": df_hop_features,
        "df_hop_features_scaled": df_hop_features_scaled,
        "hop_to_cluster_map": hop_to_cluster_map,
        "compound_oav_map": compound_oav_map,
    }


def get_target_hop_profile(df_replace_hops: pd.DataFrame, target_hop_name: str) -> pd.DataFrame:
    target_profile = df_replace_hops[df_replace_hops["Hop"] == target_hop_name]
    if target_profile.empty:
        raise ValueError(f"Target hop '{target_hop_name}' not found in df_replace_hops.")
    return target_profile[["Compound", "Concentration"]]


def get_compound_priorities(
    selected_compounds_with_weights: List[List[Any]], all_unique_compounds: np.ndarray
) -> Dict[str, float]:
    compound_priorities_raw: Dict[str, float] = {}
    for compound, weight in selected_compounds_with_weights:
        if compound not in all_unique_compounds:
            raise ValueError(f"Compound '{compound}' not found in the list of all unique compounds.")
        compound_priorities_raw[compound] = float(weight)

    sorted_priorities = sorted(
        compound_priorities_raw.items(), key=lambda item: item[1], reverse=True
    )

    return {
        compound: weight
        for compound, weight in sorted_priorities[:MAX_COMPOUNDS_TO_PRIORITIZE]
    }


def objective_function(
    hop_rates: np.ndarray,
    target_profile_df: pd.DataFrame,
    hop_compound_matrix: pd.DataFrame,
    compound_priorities_dict: Dict[str, float],
    max_concentration_factor: float,
) -> float:
    total_cost = 0.0
    for compound, weight in compound_priorities_dict.items():
        target_concentration_row = target_profile_df[
            target_profile_df["Compound"] == compound
        ]
        if not target_concentration_row.empty:
            target_concentration = (
                target_concentration_row["Concentration"].iloc[0]
                / max_concentration_factor
            )
        else:
            target_concentration = 0.0

        if compound in hop_compound_matrix.columns:
            scaled_hop_concentrations = (
                hop_compound_matrix[compound].values / max_concentration_factor
            )
            blend_concentration = np.dot(hop_rates, scaled_hop_concentrations)
        else:
            blend_concentration = 0.0

        difference = blend_concentration - target_concentration
        total_cost += difference**2 * weight

    return total_cost


def find_optimal_blend(
    target_hop_name: str,
    selected_compounds_with_weights: List[List[Any]],
    df_available_hops: pd.DataFrame,
    df_replace_hops: pd.DataFrame,
    all_unique_compounds: np.ndarray,
    max_concentration_factor: float,
) -> Dict[str, float]:
    target_profile_df = get_target_hop_profile(df_replace_hops, target_hop_name)
    compound_priorities_dict = get_compound_priorities(
        selected_compounds_with_weights, all_unique_compounds
    )

    local_hop_compound_matrix = df_available_hops.pivot_table(
        index="Hop", columns="Compound", values="Concentration"
    )
    local_hop_compound_matrix = local_hop_compound_matrix.reindex(
        columns=all_unique_compounds
    ).fillna(0)
    local_available_hops = local_hop_compound_matrix.index.tolist()

    num_hops = len(local_available_hops)
    bounds = [(0, MAX_HOP_RATE_PER_HOP) for _ in range(num_hops)]

    num_random_starts = 20
    best_result = None
    min_objective_value = np.inf

    for _ in range(num_random_starts):
        x0 = np.random.uniform(0, MAX_HOP_RATE_PER_HOP, num_hops)
        result = spo.minimize(
            fun=objective_function,
            x0=x0,
            args=(
                target_profile_df,
                local_hop_compound_matrix,
                compound_priorities_dict,
                max_concentration_factor,
            ),
            method="L-BFGS-B",
            bounds=bounds,
        )

        if result.success and result.fun < min_objective_value:
            min_objective_value = result.fun
            best_result = result

    if best_result is None:
        optimized_rates = np.zeros(num_hops)
    else:
        optimized_rates = best_result.x

    hop_rate_tuples = []
    for i, hop_name in enumerate(local_available_hops):
        if optimized_rates[i] > 1e-6:
            hop_rate_tuples.append((hop_name, optimized_rates[i]))

    hop_rate_tuples.sort(key=lambda x: x[1], reverse=True)

    if len(hop_rate_tuples) > MAX_HOPS_IN_BLEND:
        hop_rate_tuples = hop_rate_tuples[:MAX_HOPS_IN_BLEND]

    return {hop: rate for hop, rate in hop_rate_tuples}


def hop_blend_calculator(
    target_hop_name: str,
    selected_compounds_with_weights: List[List[Any]],
    df_available_hops: pd.DataFrame,
    df_replace_hops: pd.DataFrame,
    all_unique_compounds: np.ndarray,
    max_concentration_factor: float,
) -> Dict[str, Any]:
    optimal_blend = find_optimal_blend(
        target_hop_name=target_hop_name,
        selected_compounds_with_weights=selected_compounds_with_weights,
        df_available_hops=df_available_hops,
        df_replace_hops=df_replace_hops,
        all_unique_compounds=all_unique_compounds,
        max_concentration_factor=max_concentration_factor,
    )

    blend_profile = pd.Series(0.0, index=all_unique_compounds)
    local_hop_compound_matrix = df_available_hops.pivot_table(
        index="Hop", columns="Compound", values="Concentration"
    )
    local_hop_compound_matrix = local_hop_compound_matrix.reindex(
        columns=all_unique_compounds
    ).fillna(0)

    for hop_name, rate in optimal_blend.items():
        if hop_name in local_hop_compound_matrix.index:
            blend_profile += rate * local_hop_compound_matrix.loc[hop_name]

    target_profile_df = get_target_hop_profile(df_replace_hops, target_hop_name)
    target_profile = pd.Series(0.0, index=all_unique_compounds)
    for _, row in target_profile_df.iterrows():
        if row["Compound"] in target_profile.index:
            target_profile[row["Compound"]] = row["Concentration"]

    return {
        "optimal_blend": optimal_blend,
        "blend_profile": blend_profile,
        "target_profile": target_profile,
    }


def identify_cluster_defining_compounds(
    target_hop_name: str,
    df_hop_features_scaled: pd.DataFrame,
    df_hop_features: pd.DataFrame,
    hop_to_cluster_map: Dict[str, int],
    top_n: int = 10,
) -> pd.DataFrame:
    target_cluster_id = hop_to_cluster_map.get(target_hop_name)
    if target_cluster_id is None:
        return pd.DataFrame(columns=["Compound", "Mean_Diff"])

    target_cluster_hops = df_hop_features_scaled[
        df_hop_features.loc[df_hop_features_scaled.index, "Cluster"]
        == target_cluster_id
    ]
    other_hops = df_hop_features_scaled[
        df_hop_features.loc[df_hop_features_scaled.index, "Cluster"]
        != target_cluster_id
    ]

    if target_cluster_hops.empty or other_hops.empty:
        return pd.DataFrame(columns=["Compound", "Mean_Diff"])

    mean_target_cluster = target_cluster_hops.mean()
    mean_other_clusters = other_hops.mean()
    diffs = (mean_target_cluster - mean_other_clusters).abs()
    diffs = diffs.sort_values(ascending=False)

    return diffs.head(top_n).reset_index().rename(columns={"index": "Compound", 0: "Mean_Diff"})


def perform_pca_for_variance_analysis(
    df_features_scaled: pd.DataFrame, n_components: int = 3
) -> pd.DataFrame:
    if df_features_scaled.empty:
        return pd.DataFrame(columns=["Compound", "Summed_Absolute_Loading"])

    pca = PCA(n_components=n_components)
    pca.fit(df_features_scaled)

    loadings_df = pd.DataFrame(
        pca.components_,
        columns=df_features_scaled.columns,
        index=[f"PC{i+1}" for i in range(n_components)],
    )

    summed_absolute_loadings = loadings_df.abs().sum(axis=0)
    ranked_compounds = summed_absolute_loadings.sort_values(ascending=False)

    result_df = ranked_compounds.reset_index()
    result_df.columns = ["Compound", "Summed_Absolute_Loading"]
    return result_df


def build_plot_base64(comparison_df: pd.DataFrame, target_hop_name: str, plot_title_suffix: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = comparison_df[["Compound", "Target Concentration", "Blend Concentration"]].melt(
        id_vars="Compound", var_name="Profile Type", value_name="Concentration"
    )
    sns.barplot(x="Compound", y="Concentration", hue="Profile Type", data=plot_df, ax=ax, palette="viridis")
    ax.set_title(
        f"Chemical Profile Comparison: Current Blend vs. Target ({target_hop_name}){plot_title_suffix}",
        fontsize=14,
    )
    ax.set_ylabel("Concentration", fontsize=11)
    ax.set_xlabel("Chemical Compound", fontsize=11)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.xticks(rotation=45, ha="right")
    ax.legend(title="Profile", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close(fig)
    return image_base64


def create_static_scenario_report(
    state: Dict[str, Any],
    target_hop_name: str,
    selected_compounds_with_weights_list: List[List[Any]],
    current_selection_mode: str,
    blend_outside_cluster: bool,
) -> Dict[str, Any]:
    scenario_report: Dict[str, Any] = {}
    scenario_report["mode"] = current_selection_mode

    if not selected_compounds_with_weights_list:
        scenario_report["error"] = (
            "No compounds selected or weights entered. "
            "Please select items or enable 'Top 10 by Concentration' or 'Top 10 by OAV' mode."
        )
        return scenario_report

    cluster_defining_compounds_df = identify_cluster_defining_compounds(
        target_hop_name,
        state["df_hop_features_scaled"],
        state["df_hop_features"],
        state["hop_to_cluster_map"],
    )
    scenario_report["cluster_defining_compounds"] = cluster_defining_compounds_df.to_dict(
        orient="records"
    )

    pca_variance_compounds_df = perform_pca_for_variance_analysis(
        state["df_hop_features_scaled"], n_components=3
    )
    scenario_report["pca_variance_analysis"] = pca_variance_compounds_df.head(10).to_dict(
        orient="records"
    )

    if blend_outside_cluster:
        df_available_hops_for_calc = state["df_hop_results"]
        scenario_report["blend_scope_info"] = "Blending with all available hops."
    else:
        target_hop_cluster = state["hop_to_cluster_map"].get(target_hop_name)
        if target_hop_cluster is None:
            scenario_report["error"] = (
                f"Target hop '{target_hop_name}' not found in clustering map."
            )
            return scenario_report
        hops_in_target_cluster = state["df_hop_features"][
            state["df_hop_features"]["Cluster"] == target_hop_cluster
        ].index.tolist()
        df_available_hops_for_calc = state["df_hop_results"][
            state["df_hop_results"]["Hop"].isin(hops_in_target_cluster)
        ]
        if df_available_hops_for_calc.empty:
            scenario_report["error"] = (
                f"No available hops found in the same cluster ({target_hop_cluster}) as '{target_hop_name}'."
            )
            return scenario_report
        scenario_report["blend_scope_info"] = (
            f"Blending with hops from cluster {target_hop_cluster}."
        )

    calculation_results = hop_blend_calculator(
        target_hop_name=target_hop_name,
        selected_compounds_with_weights=selected_compounds_with_weights_list,
        df_available_hops=df_available_hops_for_calc,
        df_replace_hops=state["df_replace_hops"],
        all_unique_compounds=state["all_unique_compounds"],
        max_concentration_factor=state["max_concentration_factor"],
    )

    scenario_report["optimal_blend"] = calculation_results["optimal_blend"]

    target_profile_series = get_target_hop_profile(
        state["df_replace_hops"], target_hop_name
    ).set_index("Compound")["Concentration"]

    current_blend_profile = pd.Series(0.0, index=state["all_unique_compounds"])
    local_hop_compound_matrix = df_available_hops_for_calc.pivot_table(
        index="Hop", columns="Compound", values="Concentration"
    )
    local_hop_compound_matrix = local_hop_compound_matrix.reindex(
        columns=state["all_unique_compounds"]
    ).fillna(0)
    for hop_name, rate in calculation_results["optimal_blend"].items():
        if hop_name in local_hop_compound_matrix.index:
            current_blend_profile += rate * local_hop_compound_matrix.loc[hop_name]

    comparison_data = []
    plot_title_suffix = ""
    if current_selection_mode == "Prioritize by Class":
        target_class_concentrations = {cls: 0.0 for cls in state["unique_chemical_classes"]}
        blend_class_concentrations = {cls: 0.0 for cls in state["unique_chemical_classes"]}
        for compound, _ in selected_compounds_with_weights_list:
            compound_class = state["compound_to_class_map"].get(compound, "Miscellaneous")
            target_concentration = target_profile_series.get(compound, 0.0)
            blend_concentration = current_blend_profile.get(compound, 0.0)
            target_class_concentrations[compound_class] += target_concentration
            blend_class_concentrations[compound_class] += blend_concentration

        prioritized_classes_for_display = sorted(
            list(
                set(
                    state["compound_to_class_map"].get(c, "Miscellaneous")
                    for c, _ in selected_compounds_with_weights_list
                )
            )
        )
        for class_name in prioritized_classes_for_display:
            comparison_data.append(
                {
                    "Compound": class_name,
                    "Target Concentration": target_class_concentrations.get(class_name, 0.0),
                    "Blend Concentration": blend_class_concentrations.get(class_name, 0.0),
                }
            )
        plot_title_suffix = " (by Class)"
    else:
        for compound, _ in selected_compounds_with_weights_list:
            comparison_data.append(
                {
                    "Compound": compound,
                    "Target Concentration": target_profile_series.get(compound, 0.0),
                    "Blend Concentration": current_blend_profile.get(compound, 0.0),
                }
            )
        if current_selection_mode == "Top 10 by Concentration":
            plot_title_suffix = " (Top 10 by Concentration)"
        elif current_selection_mode == "Top 10 by OAV":
            plot_title_suffix = " (Top 10 by OAV)"

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df["Numeric Percentage Difference"] = comparison_df.apply(
        lambda row: (
            (row["Blend Concentration"] - row["Target Concentration"])
            / row["Target Concentration"]
            * 100
        )
        if row["Target Concentration"] != 0
        else (100 if row["Blend Concentration"] != 0 else 0),
        axis=1,
    )
    comparison_df["Percentage Difference"] = comparison_df[
        "Numeric Percentage Difference"
    ].apply(lambda x: f"{x:.2f}%")
    comparison_df["Absolute Difference"] = (
        comparison_df["Target Concentration"] - comparison_df["Blend Concentration"]
    ).abs()
    comparison_df = comparison_df.sort_values(
        by="Absolute Difference", ascending=False
    ).drop(columns=["Absolute Difference"])

    scenario_report["comparison"] = comparison_df.to_dict(orient="records")
    scenario_report["plot_image_base64"] = build_plot_base64(
        comparison_df, target_hop_name, plot_title_suffix
    )
    scenario_report["total_blend_rate"] = sum(
        calculation_results["optimal_blend"].values()
    )

    return scenario_report


def build_selected_compounds(
    state: Dict[str, Any],
    target_hop_name: str,
    current_selection_mode: str,
    selected_compounds: List[str],
    selected_classes: List[str],
) -> List[List[Any]]:
    selected_compounds_with_weights: List[List[Any]] = []
    if current_selection_mode == "Prioritize by Compound":
        items_to_prioritize_raw = selected_compounds
        num_items_to_prioritize = min(
            len(items_to_prioritize_raw), MAX_COMPOUNDS_TO_PRIORITIZE
        )
        for i in range(num_items_to_prioritize):
            item = items_to_prioritize_raw[i]
            weight = (MAX_COMPOUNDS_TO_PRIORITIZE - i) ** 2
            selected_compounds_with_weights.append([item, float(weight)])
    elif current_selection_mode == "Prioritize by Class":
        items_to_prioritize_raw = selected_classes
        class_to_compounds = {
            cls: [
                comp
                for comp, c_cls in state["compound_to_class_map"].items()
                if c_cls == cls
            ]
            for cls in state["unique_chemical_classes"]
        }
        compound_priority_for_calc: Dict[str, float] = {}
        num_items_to_prioritize_classes = min(
            len(items_to_prioritize_raw), MAX_COMPOUNDS_TO_PRIORITIZE
        )
        for i in range(num_items_to_prioritize_classes):
            class_name = items_to_prioritize_raw[i]
            weight = (MAX_COMPOUNDS_TO_PRIORITIZE - i) ** 2
            for compound in class_to_compounds.get(class_name, []):
                if compound not in compound_priority_for_calc or weight > compound_priority_for_calc[compound]:
                    compound_priority_for_calc[compound] = weight
        selected_compounds_with_weights = list(compound_priority_for_calc.items())
        selected_compounds_with_weights.sort(key=lambda x: x[1], reverse=True)
    elif current_selection_mode == "Top 10 by Concentration":
        target_profile_df = get_target_hop_profile(
            state["df_replace_hops"], target_hop_name
        )
        concentration_series = target_profile_df.set_index("Compound")["Concentration"]
        concentration_series = concentration_series[
            concentration_series.index.isin(state["all_unique_compounds"])
        ]
        sorted_compounds_by_concentration = concentration_series.nlargest(
            MAX_COMPOUNDS_TO_PRIORITIZE
        ).index.tolist()
        for i, compound in enumerate(sorted_compounds_by_concentration):
            weight = (MAX_COMPOUNDS_TO_PRIORITIZE - i) ** 2
            selected_compounds_with_weights.append([compound, float(weight)])
    elif current_selection_mode == "Top 10 by OAV":
        target_profile_df = get_target_hop_profile(
            state["df_replace_hops"], target_hop_name
        )
        compound_oav_scores = []
        for compound in state["all_unique_compounds"]:
            target_conc = (
                target_profile_df[
                    target_profile_df["Compound"] == compound
                ]["Concentration"].iloc[0]
                if compound in target_profile_df["Compound"].values
                else 0.0
            )
            oav = state["compound_oav_map"].get(compound, 0)
            if target_conc > 0 and oav > 0:
                compound_oav_scores.append((compound, oav))
        sorted_compounds_by_oav = sorted(
            compound_oav_scores, key=lambda x: x[1], reverse=True
        )
        top_oav_compounds_names = [
            item[0] for item in sorted_compounds_by_oav[:MAX_COMPOUNDS_TO_PRIORITIZE]
        ]
        for i, compound in enumerate(top_oav_compounds_names):
            weight = (MAX_COMPOUNDS_TO_PRIORITIZE - i) ** 2
            selected_compounds_with_weights.append([compound, float(weight)])

    return selected_compounds_with_weights


def parse_csv_entry(raw_value: str) -> List[str]:
    if not raw_value:
        return []
    return [entry.strip() for entry in raw_value.split(",") if entry.strip()]


def _format_report(report: Dict[str, Any]) -> str:
    if report.get("error"):
        return f"{report['mode']}: {report['error']}\n"

    lines = [f"=== {report['mode']} ===", report.get("blend_scope_info", "")] 
    optimal_blend = report.get("optimal_blend", {})
    if optimal_blend:
        lines.append("Optimal Blend:")
        for hop, rate in optimal_blend.items():
            lines.append(f"  - {hop}: {rate:.2f} lbs/bbl")
    else:
        lines.append("No hops recommended.")
    lines.append(f"Total Blend Rate: {report.get('total_blend_rate', 0):.2f} lbs/bbl")

    lines.append("\nCluster-defining compounds:")
    for row in report.get("cluster_defining_compounds", [])[:10]:
        lines.append(f"  - {row['Compound']}: {row['Mean_Diff']:.2f}")

    lines.append("\nPCA variance drivers:")
    for row in report.get("pca_variance_analysis", [])[:10]:
        lines.append(f"  - {row['Compound']}: {row['Summed_Absolute_Loading']:.2f}")

    lines.append("\nComparison (top 10 differences):")
    for row in report.get("comparison", [])[:10]:
        lines.append(
            f"  - {row['Compound']}: target {row['Target Concentration']:.2e}, "
            f"blend {row['Blend Concentration']:.2e}, diff {row['Percentage Difference']}"
        )

    return "\n".join(lines) + "\n\n"


class HopBlendApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Hop Blend Calculator")
        self.state: Dict[str, Any] | None = None
        self.plot_images: List[tk.PhotoImage] = []

        self._build_ui()

    def _build_ui(self) -> None:
        self.root.geometry("1100x800")
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        data_frame = ttk.LabelFrame(main_frame, text="1) Connect to Data", padding=12)
        data_frame.pack(fill=tk.X)

        self.source_var = tk.StringVar(value="local")
        local_radio = ttk.Radiobutton(
            data_frame, text="Local Excel file", variable=self.source_var, value="local"
        )
        sharepoint_radio = ttk.Radiobutton(
            data_frame, text="SharePoint URL", variable=self.source_var, value="sharepoint"
        )
        local_radio.grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        sharepoint_radio.grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)

        self.file_path_var = tk.StringVar()
        ttk.Label(data_frame, text="Excel file path:").grid(row=1, column=0, sticky=tk.W)
        self.file_entry = ttk.Entry(data_frame, textvariable=self.file_path_var, width=60)
        self.file_entry.grid(row=1, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Button(data_frame, text="Browse", command=self._browse_file).grid(
            row=1, column=2, padx=4, pady=4
        )

        self.sharepoint_url_var = tk.StringVar(value=os.getenv(DATA_ENV_SHAREPOINT_URL, ""))
        self.sharepoint_token_var = tk.StringVar(value=os.getenv(DATA_ENV_SHAREPOINT_TOKEN, ""))

        ttk.Label(data_frame, text="SharePoint URL:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(data_frame, textvariable=self.sharepoint_url_var, width=60).grid(
            row=2, column=1, sticky=tk.W, padx=4, pady=4
        )
        ttk.Label(data_frame, text="SharePoint token (optional):").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(data_frame, textvariable=self.sharepoint_token_var, width=60, show="*").grid(
            row=3, column=1, sticky=tk.W, padx=4, pady=4
        )

        ttk.Button(data_frame, text="Connect Data", command=self._connect_data).grid(
            row=4, column=1, sticky=tk.W, padx=4, pady=6
        )
        self.status_label = ttk.Label(data_frame, text="Status: Not connected", foreground="#b42318")
        self.status_label.grid(row=4, column=2, sticky=tk.W)

        calc_frame = ttk.LabelFrame(main_frame, text="2) Configure Calculation", padding=12)
        calc_frame.pack(fill=tk.X, pady=10)

        ttk.Label(calc_frame, text="Target Hop:").grid(row=0, column=0, sticky=tk.W)
        self.target_hop_var = tk.StringVar()
        self.target_hop_combo = ttk.Combobox(calc_frame, textvariable=self.target_hop_var, width=40, state="disabled")
        self.target_hop_combo.grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)

        ttk.Label(calc_frame, text="Modes:").grid(row=1, column=0, sticky=tk.NW)
        self.mode_vars: Dict[str, tk.BooleanVar] = {
            "Prioritize by Compound": tk.BooleanVar(value=True),
            "Prioritize by Class": tk.BooleanVar(),
            "Top 10 by Concentration": tk.BooleanVar(),
            "Top 10 by OAV": tk.BooleanVar(),
        }
        modes_frame = ttk.Frame(calc_frame)
        modes_frame.grid(row=1, column=1, sticky=tk.W)
        for idx, (label, var) in enumerate(self.mode_vars.items()):
            ttk.Checkbutton(modes_frame, text=label, variable=var).grid(
                row=idx, column=0, sticky=tk.W
            )

        ttk.Label(calc_frame, text="Compounds (comma-separated):").grid(row=2, column=0, sticky=tk.W)
        self.compounds_var = tk.StringVar()
        self.compounds_entry = ttk.Entry(calc_frame, textvariable=self.compounds_var, width=60, state="disabled")
        self.compounds_entry.grid(row=2, column=1, sticky=tk.W, padx=4, pady=4)

        ttk.Label(calc_frame, text="Classes (comma-separated):").grid(row=3, column=0, sticky=tk.W)
        self.classes_var = tk.StringVar()
        self.classes_entry = ttk.Entry(calc_frame, textvariable=self.classes_var, width=60, state="disabled")
        self.classes_entry.grid(row=3, column=1, sticky=tk.W, padx=4, pady=4)

        self.blend_outside_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            calc_frame,
            text="Blend outside of target hop's cluster",
            variable=self.blend_outside_var,
        ).grid(row=4, column=1, sticky=tk.W, padx=4, pady=4)

        self.calculate_button = ttk.Button(calc_frame, text="Calculate Blend", command=self._calculate, state="disabled")
        self.calculate_button.grid(row=5, column=1, sticky=tk.W, padx=4, pady=6)

        results_frame = ttk.LabelFrame(main_frame, text="Results", padding=12)
        results_frame.pack(fill=tk.BOTH, expand=True)
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=16)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        self.image_frame = ttk.Frame(results_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=8)

    def _browse_file(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select hop Excel file",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*")],
        )
        if filename:
            self.file_path_var.set(filename)

    def _connect_data(self) -> None:
        self.status_label.configure(text="Status: Connecting...", foreground="#4a5568")
        self.root.update_idletasks()

        try:
            sharepoint_url = None
            token = None
            source_path = None
            if self.source_var.get() == "sharepoint":
                sharepoint_url = self.sharepoint_url_var.get().strip()
                token = self.sharepoint_token_var.get().strip() or None
                if not sharepoint_url:
                    raise DataLoadError("SharePoint URL is required when SharePoint mode is selected.")
            else:
                source_path = self.file_path_var.get().strip() or None

            data_bytes = _load_excel_bytes(source_path, sharepoint_url, token)
            self.state = build_state(data_bytes)
            target_hops = sorted(self.state["df_replace_hops"]["Hop"].unique().tolist())
            self.target_hop_combo["values"] = target_hops
            if target_hops:
                self.target_hop_combo.current(0)

            self.target_hop_combo.configure(state="readonly")
            self.compounds_entry.configure(state="normal")
            self.classes_entry.configure(state="normal")
            self.calculate_button.configure(state="normal")
            self.status_label.configure(text="Status: Connected", foreground="#2f855a")
        except Exception as exc:
            messagebox.showerror("Data connection failed", str(exc))
            self.status_label.configure(text="Status: Not connected", foreground="#b42318")

    def _clear_results(self) -> None:
        self.results_text.delete("1.0", tk.END)
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        self.plot_images.clear()

    def _calculate(self) -> None:
        if not self.state:
            messagebox.showwarning("Connect data", "Connect to a data source first.")
            return

        selected_modes = [mode for mode, var in self.mode_vars.items() if var.get()]
        if not selected_modes:
            messagebox.showwarning("Select modes", "Select at least one prioritization mode.")
            return

        target_hop = self.target_hop_var.get()
        selected_compounds = parse_csv_entry(self.compounds_var.get())
        selected_classes = parse_csv_entry(self.classes_var.get())

        self._clear_results()

        for mode in selected_modes:
            selected_compounds_with_weights = build_selected_compounds(
                self.state,
                target_hop,
                mode,
                selected_compounds,
                selected_classes,
            )
            report = create_static_scenario_report(
                self.state,
                target_hop,
                selected_compounds_with_weights,
                mode,
                self.blend_outside_var.get(),
            )
            self.results_text.insert(tk.END, _format_report(report))

            if report.get("plot_image_base64"):
                try:
                    photo = tk.PhotoImage(data=report["plot_image_base64"])
                    self.plot_images.append(photo)
                    label = ttk.Label(self.image_frame, image=photo)
                    label.pack(pady=6)
                except tk.TclError:
                    self.results_text.insert(
                        tk.END,
                        "Plot image could not be rendered in the UI."
                        " You can still access values in the text report.\n",
                    )


if __name__ == "__main__":
    root = tk.Tk()
    app = HopBlendApp(root)
    root.mainloop()
