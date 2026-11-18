from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.affinity import scale as shp_scale
from shapely.geometry import Point
from shapely.wkt import loads as wkt_loads
from sklearn.metrics import adjusted_rand_score

from tools.cli.compartments import main as compartment_cli_main
from tools.decomap.api import _boundary_to_polygon, _load_decomap_geometries, run_decomap_from_bundle, run_decomap_from_pkl
from tools.decomap.core import compute_exact_csr_assortativity, run_decomap_core
from tools.decomap.naming import annotate_compartments, load_organelle_markers
from tools.decomap.regions import build_compartment_regions
from tools.engines.coloc_region_annotation import CellGeometry


def _make_structured_cells(
    n_cells: int = 12,
    seed: int = 2026,
    scale_factor: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame], np.ndarray]:
    """Synthesize cells with Nuclear, Perinuclear-ER, Cytosolic, and rare Foci microdomains."""
    rng = np.random.default_rng(seed)
    rows = []
    gt_domains = []
    cell_bounds: dict[str, pd.DataFrame] = {}
    nuc_bounds: dict[str, pd.DataFrame] = {}

    nuc_genes = ["MALAT1", "NEAT1", "SPPL2B"]
    er_genes = ["CALR", "HSPA5", "COL1A1"]
    cyto_genes = ["GAPDH", "ACTB", "RAB13", "PFN1"]
    foci_genes = ["SNHG1", "RBM5"]

    for c_idx in range(n_cells):
        cid = f"cell_{c_idx}"
        ox = (c_idx % 4) * 260.0 * scale_factor
        oy = (c_idx // 4) * 260.0 * scale_factor
        r_cell = 90.0 * scale_factor
        r_nuc = 36.0 * scale_factor

        c_poly = Point(ox, oy).buffer(r_cell, quad_segs=32)
        n_poly = Point(ox, oy).buffer(r_nuc, quad_segs=32)
        cx_b, cy_b = c_poly.exterior.xy
        nx_b, ny_b = n_poly.exterior.xy
        cell_bounds[cid] = pd.DataFrame({"x": list(cx_b), "y": list(cy_b)})
        nuc_bounds[cid] = pd.DataFrame({"x": list(nx_b), "y": list(ny_b)})

        # 0: Nuclear (inside r_nuc)
        for _ in range(90):
            ang = rng.uniform(0, 2 * np.pi)
            rr = r_nuc * 0.92 * np.sqrt(rng.uniform(0, 1))
            g = rng.choice(nuc_genes, p=[0.45, 0.35, 0.20])
            rows.append((cid, g, ox + rr * np.cos(ang), oy + rr * np.sin(ang)))
            gt_domains.append(0)

        # 1: Perinuclear-ER ring (r_nuc .. 1.45 * r_nuc)
        for _ in range(90):
            ang = rng.uniform(0, 2 * np.pi)
            rr = rng.uniform(r_nuc * 1.04, r_nuc * 1.45)
            g = rng.choice(er_genes)
            rows.append((cid, g, ox + rr * np.cos(ang), oy + rr * np.sin(ang)))
            gt_domains.append(1)

        # 3: Rare dense Foci microdomain (~5% of spots) placed first so spatial domains are disjoint
        f_ang = rng.uniform(0, 2 * np.pi)
        fx = ox + 0.72 * r_cell * np.cos(f_ang)
        fy = oy + 0.72 * r_cell * np.sin(f_ang)
        for _ in range(18):
            dx, dy = rng.normal(0.0, 0.025 * r_cell, size=2)
            g = rng.choice(foci_genes)
            rows.append((cid, g, fx + dx, fy + dy))
            gt_domains.append(3)

        # 2: Bulk Cytosolic / Peripheral (1.52 * r_nuc .. 0.95 * r_cell, outside foci core)
        n_cyto = 0
        while n_cyto < 140:
            ang = rng.uniform(0, 2 * np.pi)
            rr = rng.uniform(r_nuc * 1.52, r_cell * 0.95)
            px, py = ox + rr * np.cos(ang), oy + rr * np.sin(ang)
            if np.hypot(px - fx, py - fy) < 0.08 * r_cell:
                continue
            g = rng.choice(cyto_genes)
            rows.append((cid, g, px, py))
            gt_domains.append(2)
            n_cyto += 1

    df = pd.DataFrame(rows, columns=["cell", "gene", "x", "y"])
    return df, cell_bounds, nuc_bounds, np.asarray(gt_domains, dtype=int)


def test_closed_form_csr_moments_match_permutations():
    """Verify exact closed-form CSR expectation and SD match 5,000 finite label permutations."""
    rng = np.random.default_rng(77)
    n_spots = 110
    xy = rng.normal(0.0, 25.0, size=(n_spots, 2))
    gene_codes = rng.choice(6, size=n_spots, p=[0.35, 0.25, 0.15, 0.12, 0.08, 0.05]).astype(np.int32)

    dmat = cKDTree(xy).sparse_distance_matrix(cKDTree(xy), 18.0, output_type="coo_matrix")
    off = dmat.row != dmat.col
    rr, cc = dmat.row[off], dmat.col[off]
    ww = np.exp(-0.5 * (dmat.data[off] / 9.0) ** 2)

    _, mu_exact, sd_exact, _, _ = compute_exact_csr_assortativity(rr, cc, ww, gene_codes, 6)

    upper = rr < cc
    ru, cu, wu = rr[upper], cc[upper], ww[upper]
    sim_vals = np.empty(5000, dtype=np.float64)
    for it in range(5000):
        perm = rng.permutation(gene_codes)
        sim_vals[it] = float(wu[perm[ru] == perm[cu]].sum())

    assert np.isclose(mu_exact, sim_vals.mean(), rtol=0.015)
    assert np.isclose(sd_exact, sim_vals.std(ddof=1), rtol=0.03)


def test_csr_negative_control_blocks_false_compartments():
    """Under pure intracellular CSR, DeCoMap flags >=90% of cells as Unstructured."""
    rng = np.random.default_rng(99)
    df, cell_bounds, _, _ = _make_structured_cells(n_cells=25, seed=99)
    # Permute gene labels independently within each cell to destroy all spatial compartments
    csr_df = df.copy()
    for cid, idx in csr_df.groupby("cell").indices.items():
        csr_df.iloc[idx, csr_df.columns.get_loc("gene")] = rng.permutation(
            csr_df.iloc[idx]["gene"].to_numpy()
        )

    res = run_decomap_from_bundle(
        {"data_df": csr_df, "cell_boundary": cell_bounds, "nuclear_boundary": {}},
        n_clusters=4,
        calibrate_csr=True,
        fdr_alpha=0.05,
        random_state=42,
    )
    false_structured_rate = res.n_structured_cells / res.n_total_cells
    assert false_structured_rate <= 0.10
    assert (res.point_df["compartment_name"] == "Unstructured").mean() >= 0.90


def test_scale_invariance_and_zero_nuclear_boundary_leakage():
    """Verify 100x coordinate scaling invariance and exact invariance to nuclear_boundary presence."""
    df_1x, cb_1x, nb_1x, gt = _make_structured_cells(n_cells=10, seed=42, scale_factor=1.0)
    df_100x, cb_100x, nb_100x, _ = _make_structured_cells(n_cells=10, seed=42, scale_factor=100.0)

    # 1. Run WITH vs WITHOUT nuclear_boundary in bundle -> assignments must be 100% identical!
    res_with_nuc = run_decomap_from_bundle(
        {"data_df": df_1x, "cell_boundary": cb_1x, "nuclear_boundary": nb_1x},
        n_clusters=4,
        random_state=42,
        threads=2,
    )
    res_no_nuc = run_decomap_from_bundle(
        {"data_df": df_1x, "cell_boundary": cb_1x, "nuclear_boundary": {}},
        n_clusters=4,
        random_state=42,
        threads=2,
    )
    assert np.array_equal(
        res_with_nuc.point_df["subdomain"].to_numpy(),
        res_no_nuc.point_df["subdomain"].to_numpy(),
    )
    assert "posthoc_nuclear_fraction" in res_with_nuc.compartment_summary.columns
    assert "posthoc_nuclear_fraction" not in res_no_nuc.compartment_summary.columns

    # 2. High accuracy recovery of all 4 simulated domains including rare Foci (domain 3)
    ari = adjusted_rand_score(gt, res_no_nuc.point_df["raw_subdomain"].to_numpy())
    assert ari >= 0.80
    names = set(res_no_nuc.compartment_summary["base_organelle"].tolist())
    assert "Nuclear" in names
    # The toy panel carries only 3 ER marker genes, below the naming evidence threshold.
    assert "Perinuclear-ER" not in names

    # 3. 100x scale invariance (microns vs pixels)
    res_100x = run_decomap_from_bundle(
        {"data_df": df_100x, "cell_boundary": cb_100x, "nuclear_boundary": {}},
        n_clusters=4,
        random_state=42,
        threads=2,
    )
    assert np.array_equal(
        res_no_nuc.point_df["subdomain"].to_numpy(),
        res_100x.point_df["subdomain"].to_numpy(),
    )


def test_cli_and_pkl_output_compatibility(tmp_path: Path):
    """Verify subcellfeat-compartment --engine decomap writes backward-compatible CSVs."""
    df, cb, nb, _ = _make_structured_cells(n_cells=6, seed=7)
    df["batch"] = "b0"
    pkl_path = tmp_path / "toy_data_dict.pkl"
    with pkl_path.open("wb") as fh:
        pickle.dump({"data_df": df, "cell_boundary": cb, "nuclear_boundary": nb}, fh)

    out_prefix = tmp_path / "out" / "run"
    rc = compartment_cli_main(
        [
            "--pkl",
            str(pkl_path),
            "--out-prefix",
            str(out_prefix),
            "--engine",
            "decomap",
            "--n-clusters",
            "4",
        ]
    )
    assert rc == 0
    pts_csv = Path(f"{out_prefix}_batch-b0_sampled_subdomain_points.csv")
    sum_csv = Path(f"{out_prefix}_batch-b0_compartment_summary.csv")
    assert pts_csv.exists()
    assert sum_csv.exists()
    assert Path(f"{out_prefix}_batch-b0_compartment_regions.csv").exists()
    pts_df = pd.read_csv(pts_csv)
    for required_col in ["cell", "gene", "x", "y", "subdomain", "compartment_name", "rnaflux_r", "rnaflux_g", "rnaflux_b"]:
        assert required_col in pts_df.columns


def test_naming_requires_marker_evidence(tmp_path: Path):
    """Few-gene sets are ignored, near-tied sibling sets fall back to the parent, user CSVs are honoured."""
    genes = [f"G{i}" for i in range(20)]
    scores = np.zeros((2, 20))
    scores[0, 0:8] = 3.0
    scores[1, 10:12] = 3.0
    marker_csv = tmp_path / "markers.csv"
    pd.DataFrame(
        {
            "gene": genes[0:8] + genes[0:7] + genes[10:12],
            "compartment": ["Nuclear"] * 8 + ["Nucleolar"] * 7 + ["Cytosolic"] * 2,
        }
    ).to_csv(marker_csv, index=False)

    marker_df = load_organelle_markers(marker_csv)
    assert (marker_df["weight"] == 1.0).all()
    summary = annotate_compartments(
        scores, genes, cluster_RadialFrac=np.array([0.2, 0.9]), marker_df=marker_df
    )
    assert summary["base_organelle"].tolist() == ["Nuclear", "Peripheral-Domain"]
    assert np.isnan(summary.loc[1, "z_cytosolic"])


def test_compartment_regions_recover_nucleus_area():
    """Rasterized regions stay inside each cell and the nuclear region matches the true nucleus area."""
    df, cb, nb, gt = _make_structured_cells(n_cells=8, seed=3)
    bundle = {"data_df": df, "cell_boundary": cb, "nuclear_boundary": {}}
    res = run_decomap_from_bundle(bundle, n_clusters=4, random_state=42, threads=2)
    regions = build_compartment_regions(
        res.point_df, _load_decomap_geometries(bundle), res.compartment_summary, threads=2
    )
    assert (regions.groupby("cell")["area_fraction"].sum() <= 1.0 + 1e-6).all()

    nuc_label = np.bincount(res.point_df["raw_subdomain"].to_numpy()[gt == 0]).argmax()
    dices = []
    for cid, rows in regions[regions["subdomain"] == nuc_label].groupby("cell"):
        pred = wkt_loads(rows["geometry_wkt"].iloc[0])
        true_nuc = _boundary_to_polygon(nb[cid])
        dices.append(2.0 * pred.intersection(true_nuc).area / (pred.area + true_nuc.area))
    assert len(dices) == 8
    assert np.median(dices) >= 0.85


def test_density_adaptive_scale_only_for_sparse_panels():
    """Fine scale grows to reach the per-gene neighbour target only when the median cell is too sparse."""
    df, cb, _, _ = _make_structured_cells(n_cells=8, seed=5)
    dense = run_decomap_from_bundle({"data_df": df, "cell_boundary": cb, "nuclear_boundary": {}}, n_clusters=4, threads=2)
    assert dense.scale_factor == 1.0

    sparse_df = df.groupby("cell", group_keys=False).sample(n=40, random_state=0).reset_index(drop=True)
    sparse = run_decomap_from_bundle(
        {"data_df": sparse_df, "cell_boundary": cb, "nuclear_boundary": {}}, n_clusters=3, threads=2
    )
    n_genes = sparse_df["gene"].nunique()
    m_fine = 2.0 * 40 * 0.07**2 / n_genes
    assert np.isclose(sparse.scale_factor, np.sqrt(0.088 / m_fine))
    assert np.isclose(sparse.sigma_fine_frac, 0.07 * sparse.scale_factor)
