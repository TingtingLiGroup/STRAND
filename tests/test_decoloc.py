from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from shapely.geometry import Point

from tools.decoloc.api import run_decoloc_from_bundle, summarize_negative_controls
from tools.decoloc.core import (
    assign_disjoint_transcript_regions,
    compute_cell_region_thresholds,
    run_decoloc_core,
)
from tools.engines.coloc_region_annotation import CellGeometry


def _circle_geom(cell_r: float = 100.0, nuc_r: float = 50.0, cid: str = "c0") -> CellGeometry:
    return CellGeometry(
        cell_id=cid,
        cell=Point(0.0, 0.0).buffer(cell_r, quad_segs=32),
        nuclear=Point(0.0, 0.0).buffer(nuc_r, quad_segs=32),
    )


def test_closed_form_expectations_match_monte_carlo_permutations():
    """Verify exact closed-form E_cell and E_comp match explicit label permutations."""
    rng = np.random.default_rng(123)
    geom = _circle_geom(100.0, 50.0, "c0")

    # Sample 90 spots inside the disk of radius 95 with 5 genes of varying abundance.
    angles = rng.uniform(0.0, 2.0 * np.pi, size=90)
    radii = 95.0 * np.sqrt(rng.uniform(0.0, 1.0, size=90))
    xy = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    genes = rng.choice(["G0", "G1", "G2", "G3", "G4"], size=90, p=[0.35, 0.25, 0.20, 0.12, 0.08])

    # Run on a single cell (min_coexpr_cells=1) with Gaussian kernel.
    df = pd.DataFrame({"cell": "c0", "gene": genes, "x": xy[:, 0], "y": xy[:, 1]})
    radius = 22.0
    sigma = 0.5 * radius
    res = run_decoloc_core(
        df,
        geometries={"c0": geom},
        distance=radius,
        distance_mode="physical",
        region_threshold_mode="proportional",
        kernel="gaussian",
        kernel_sigma_frac=0.5,
        min_coexpr_cells=1,
        threads=1,
    )

    t_nuc, t_cn, t_cp = compute_cell_region_thresholds(geom, mode="proportional")
    regs = assign_disjoint_transcript_regions(
        xy, geom, t_nucleus=t_nuc, t_cyto_nuclear=t_cn, t_cyto_peri=t_cp
    )
    # Disjoint tag check: every spot has exactly one valid region in 0..3.
    assert set(np.unique(regs)).issubset({0, 1, 2, 3})

    # Precompute pairwise Gaussian weights on fixed spot graph.
    pairs = cKDTree(xy).query_pairs(r=radius, output_type="ndarray")
    dists = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
    w_edges = np.exp(-0.5 * (dists / sigma) ** 2)

    # Monte Carlo conditioned permutations for top pair (G0, G1).
    row = res.all_pairs[
        (res.all_pairs["gene_1"] == "G0") & (res.all_pairs["gene_2"] == "G1")
    ].iloc[0]

    def _mc_expected(stratified: bool, n_perm: int = 8000) -> float:
        vals = []
        for anchor, target in [("G0", "G1"), ("G1", "G0")]:
            is_anchor = genes == anchor
            non_anchor_idx = np.flatnonzero(~is_anchor)
            non_anchor_labels = genes[non_anchor_idx]
            acc = 0.0
            for _ in range(n_perm):
                perm_genes = genes.copy()
                if not stratified:
                    perm_genes[non_anchor_idx] = rng.permutation(non_anchor_labels)
                else:
                    for v in range(4):
                        mask_v = (~is_anchor) & (regs == v)
                        idx_v = np.flatnonzero(mask_v)
                        if len(idx_v) > 1:
                            perm_genes[idx_v] = rng.permutation(genes[idx_v])
                gp = perm_genes[pairs[:, 0]]
                gq = perm_genes[pairs[:, 1]]
                hit = ((gp == anchor) & (gq == target)) | ((gp == target) & (gq == anchor))
                acc += float(w_edges[hit].sum())
            vals.append(acc / n_perm)
        return 0.5 * (vals[0] + vals[1])

    mc_cell = _mc_expected(stratified=False)
    mc_comp = _mc_expected(stratified=True)

    assert np.isclose(row["exp_cell"], mc_cell, rtol=0.02, atol=0.03)
    assert np.isclose(row["exp_comp"], mc_comp, rtol=0.02, atol=0.03)


def test_sparse_variance_identity_matches_dense_cell_by_cell_loop():
    """Verify sparse matrix second-moment expansion matches naive cell-by-cell paired t-test to < 1e-12."""
    rng = np.random.default_rng(99)
    n_cells = 25
    geoms = {}
    rows = []
    gene_pool = [f"G{i}" for i in range(8)]

    for c in range(n_cells):
        cid = f"cell_{c:02d}"
        rc = float(rng.uniform(80.0, 120.0))
        geoms[cid] = _circle_geom(rc, 0.5 * rc, cid)
        n_spots = int(rng.integers(25, 70))
        ang = rng.uniform(0, 2 * np.pi, size=n_spots)
        rad = (0.92 * rc) * np.sqrt(rng.uniform(0, 1, size=n_spots))
        # Let different cells express different subsets of genes so zero-edge co-expressing cells exist.
        active = rng.choice(gene_pool, size=int(rng.integers(4, 8)), replace=False)
        g_labels = rng.choice(active, size=n_spots)
        for x, y, g in zip(rad * np.cos(ang), rad * np.sin(ang), g_labels):
            rows.append({"cell": cid, "gene": g, "x": float(x), "y": float(y)})

    df = pd.DataFrame(rows)
    res = run_decoloc_core(
        df,
        geometries=geoms,
        distance=18.0,
        distance_mode="physical",
        region_threshold_mode="proportional",
        kernel="uniform",
        min_coexpr_cells=5,
        threads=2,
    )

    # Naive cell-by-cell reference computation for every active pair in res.all_pairs.
    per_cell = {}
    for cid, sub in df.groupby("cell"):
        xy = sub[["x", "y"]].to_numpy(float)
        gs = sub["gene"].to_numpy(str)
        geom = geoms[cid]
        tn, tcn, tcp = compute_cell_region_thresholds(geom, mode="proportional")
        regs = assign_disjoint_transcript_regions(xy, geom, t_nucleus=tn, t_cyto_nuclear=tcn, t_cyto_peri=tcp)
        pairs = cKDTree(xy).query_pairs(r=18.0, output_type="ndarray")
        if len(pairs):
            het = gs[pairs[:, 0]] != gs[pairs[:, 1]]
            pairs = pairs[het]
        per_cell[cid] = (gs, regs, pairs)

    for _, row in res.all_pairs.iterrows():
        g1, g2 = row["gene_1"], row["gene_2"]
        r_cell_list = []
        r_comp_list = []
        for cid, (gs, regs, pairs) in per_cell.items():
            n1 = int(np.sum(gs == g1))
            n2 = int(np.sum(gs == g2))
            if n1 == 0 or n2 == 0:
                continue
            ntot = len(gs)
            obs = 0.0
            d1_v = np.zeros(4)
            d2_v = np.zeros(4)
            for p, q in pairs:
                gp, gq = gs[p], gs[q]
                if (gp == g1 and gq == g2) or (gp == g2 and gq == g1):
                    obs += 1.0
                if gp == g1:
                    d1_v[regs[q]] += 1.0
                elif gq == g1:
                    d1_v[regs[p]] += 1.0
                if gp == g2:
                    d2_v[regs[q]] += 1.0
                elif gq == g2:
                    d2_v[regs[p]] += 1.0

            e_cell = 0.5 * (
                (d1_v.sum() * n2 / (ntot - n1)) + (d2_v.sum() * n1 / (ntot - n2))
            )
            e_comp = 0.0
            for v in range(4):
                nv_tot = int(np.sum(regs == v))
                nv1 = int(np.sum((gs == g1) & (regs == v)))
                nv2 = int(np.sum((gs == g2) & (regs == v)))
                term1 = (d1_v[v] * nv2 / (nv_tot - nv1)) if (nv_tot > nv1) else 0.0
                term2 = (d2_v[v] * nv1 / (nv_tot - nv2)) if (nv_tot > nv2) else 0.0
                e_comp += 0.5 * (term1 + term2)

            r_cell_list.append(obs - e_cell)
            r_comp_list.append(obs - e_comp)

        r_cell_arr = np.asarray(r_cell_list)
        r_comp_arr = np.asarray(r_comp_list)
        assert len(r_cell_arr) == int(row["n_coexpr_cells"])

        ref_t_cell = r_cell_arr.mean() / (r_cell_arr.std(ddof=1) / np.sqrt(len(r_cell_arr)))
        ref_t_comp = r_comp_arr.mean() / (r_comp_arr.std(ddof=1) / np.sqrt(len(r_comp_arr)))
        ref_p_cell = stats.t.sf(ref_t_cell, df=len(r_cell_arr) - 1)
        ref_p_comp = stats.t.sf(ref_t_comp, df=len(r_comp_arr) - 1)

        assert np.isclose(row["t_cell"], ref_t_cell, rtol=1e-11, atol=1e-12)
        assert np.isclose(row["t_comp"], ref_t_comp, rtol=1e-11, atol=1e-12)
        assert np.isclose(row["p_cell"], ref_p_cell, rtol=1e-11, atol=1e-12)
        assert np.isclose(row["p_comp"], ref_p_comp, rtol=1e-11, atol=1e-12)


def test_homotypic_foci_do_not_corrupt_heterotypic_test_and_boundaryless_mode():
    """Injecting dense homotypic foci of FOCI_GENE must not create false positives or shift B-C expectations."""
    rng = np.random.default_rng(7)
    rows_base = []
    rows_with_foci = []
    geoms = {}

    for c in range(30):
        cid = f"c_{c}"
        geoms[cid] = _circle_geom(100.0, 50.0, cid)
        for g in ["A_rand", "B_rand", "NegPrb1"]:
            pts = rng.uniform(-60.0, 60.0, size=(12, 2))
            for x, y in pts:
                rec = {"cell": cid, "gene": g, "x": float(x), "y": float(y)}
                rows_base.append(rec)
                rows_with_foci.append(rec)
        # Add 80 ultra-dense self-clustered transcripts of FOCI_GENE at a single point!
        center = rng.uniform(-40.0, 40.0, size=2)
        foci_pts = center + rng.normal(0.0, 0.2, size=(80, 2))
        for x, y in foci_pts:
            rows_with_foci.append({"cell": cid, "gene": "FOCI_GENE", "x": float(x), "y": float(y)})

    res_foci = run_decoloc_core(
        pd.DataFrame(rows_with_foci),
        geometries=geoms,
        distance=12.0,
        distance_mode="physical",
        min_coexpr_cells=10,
        threads=1,
    )
    # Neither FOCI_GENE nor NegPrb1 should trigger false significant colocalization.
    assert len(res_foci.significant_pairs) == 0

    neg_summary = summarize_negative_controls(res_foci.all_pairs, genes=res_foci.genes)
    assert not neg_summary.empty
    assert (neg_summary["n_control_associated_pairs"] == 0).all()

    # Also test boundary-less spatial tile fallback via run_decoloc_from_bundle.
    df_no_cell = pd.DataFrame(rows_base).drop(columns=["cell"])
    res_grid, _ = run_decoloc_from_bundle(
        {"data_df": df_no_cell},
        distance=12.0,
        grid_tile_size=50.0,
        min_coexpr_cells=2,
        min_gene_cells=2,
        threads=1,
    )
    assert not res_grid.has_compartment_null
    assert len(res_grid.significant_pairs) == 0


def test_subcellfeat_coloc_cli_engine_decoloc_dispatch(tmp_path):
    """Verify subcellfeat-coloc --engine decoloc runs end-to-end and writes direct_micro CSV."""
    import pickle
    import subprocess
    import sys

    rng = np.random.default_rng(11)
    rows, cb, nb = [], {}, {}
    for c in range(20):
        cid = f"c_{c:02d}"
        g = _circle_geom(100.0, 50.0, cid)
        cb[cid] = np.asarray(g.cell.exterior.coords, dtype=float)
        nb[cid] = np.asarray(g.nuclear.exterior.coords, dtype=float)
        pts_a = rng.uniform(-35.0, 35.0, size=(10, 2))
        pts_b = pts_a + rng.normal(0.0, 1.0, size=(10, 2))
        pts_c = rng.uniform(-60.0, 60.0, size=(15, 2))
        for gn, pts in [("GA", pts_a), ("GB", pts_b), ("GC", pts_c)]:
            for x, y in pts:
                rows.append({"cell": cid, "gene": gn, "x": float(x), "y": float(y)})

    pkl_path = tmp_path / "mini.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"data_df": pd.DataFrame(rows), "cell_boundary": cb, "nuclear_boundary": nb}, f)

    out_dir = tmp_path / "out_decoloc"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.cli.colocalization",
            "--pkl",
            str(pkl_path),
            "--out-dir",
            str(out_dir),
            "--engine",
            "decoloc",
            "--decoloc-min-coexpr-cells",
            "10",
            "--threads",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    direct_df = pd.read_csv(out_dir / "decoloc_direct_micro_pairs.csv")
    assert len(direct_df) == 1
    assert set(direct_df.iloc[0][["gene_1", "gene_2"]]) == {"GA", "GB"}

