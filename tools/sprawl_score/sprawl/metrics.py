import shapely
import shapely.geometry

import multiprocessing as mp
import functools
import zlib

import pandas as pd
import numpy as np
import collections
import itertools
import random
import math
import time
import sys
import os

from . import utils


def _make_cell_rng(random_seed: int, metric: str, cell_id: str) -> np.random.Generator:
    token = f"{random_seed}:{metric}:{cell_id}".encode("utf-8")
    seed_u32 = zlib.crc32(token) & 0xFFFFFFFF
    return np.random.default_rng(seed_u32)


def _mean_pairs_dist_rng(spots: np.ndarray, num_pairs: int, rng: np.random.Generator) -> float:
    m = spots.shape[0]
    i1 = rng.integers(0, m, size=num_pairs)
    offset = rng.integers(1, m, size=num_pairs)
    i2 = (i1 + offset) % m
    diff = spots[i1] - spots[i2]
    return float(np.sqrt(np.sum(diff * diff, axis=1)).mean())


def _mean_pairs_angle_rng(
    spots: np.ndarray,
    centroid: np.ndarray,
    num_pairs: int,
    rng: np.random.Generator,
) -> float:
    m = spots.shape[0]
    i1 = rng.integers(0, m, size=num_pairs)
    offset = rng.integers(1, m, size=num_pairs)
    i2 = (i1 + offset) % m
    v1 = spots[i1] - centroid
    v2 = spots[i2] - centroid
    n1 = np.maximum(np.linalg.norm(v1, axis=1), 1e-12)
    n2 = np.maximum(np.linalg.norm(v2, axis=1), 1e-12)
    cos_ang = np.clip(np.sum(v1 * v2, axis=1) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_ang).mean())


def _test(cells, **kwargs):
    """
    Test metric
    Returns a table with all scores of 0 and variances of 1
    """
    data = {
        'metric':[],
        'cell_id':[],
        'annotation':[],
        'num_spots':[],
        'gene':[],
        'num_gene_spots':[],
        'score':[],
        'variance':[],
    }

    for cell in cells:
        for gene in cell.genes:
            data['metric'].append('test')
            data['cell_id'].append(cell.cell_id)
            data['annotation'].append(cell.annotation)
            data['num_spots'].append(cell.n)
            data['gene'].append(gene)
            data['num_gene_spots'].append(cell.gene_counts[gene])
            data['score'].append(0)
            data['variance'].append(1)

    return pd.DataFrame(data)


def _peripheral(cell, ret_spot_ranks=False):
    """
    Helper peripheral function gets called by peripheral() for multiprocessing of each cell
    """

    data = {
        'metric':[],
        'cell_id':[],
        'annotation':[],
        'num_spots':[],
        'gene':[],
        'num_gene_spots':[],
        'score':[],
        'variance':[],
    }

    periph_dists = []
    spot_genes = []

    for zslice in cell.zslices:
        z_boundary = cell.boundaries[zslice]
        z_spot_coords = np.asarray(cell.spot_coords[zslice], dtype=np.float64)
        z_spot_genes = cell.spot_genes[zslice]

        poly = shapely.geometry.Polygon(z_boundary)
        if len(z_spot_coords) > 0:
            dists = shapely.distance(poly.boundary, shapely.points(z_spot_coords))
            periph_dists.extend(dists.tolist())
            spot_genes.extend(z_spot_genes)

    spot_genes = np.array(spot_genes)
    spot_ranks = np.array(periph_dists).argsort().argsort()+1

    if ret_spot_ranks:
        return spot_genes,spot_ranks

    exp_med = (cell.n+1)/2
    for gene in cell.genes:
        gene_ranks = spot_ranks[spot_genes == gene]
        obs_med = np.median(gene_ranks)
        score = (exp_med-obs_med)/(exp_med-1)

        data['metric'].append('periph')
        data['cell_id'].append(cell.cell_id)
        data['annotation'].append(cell.annotation)
        data['num_spots'].append(cell.n)
        data['gene'].append(gene)
        data['num_gene_spots'].append(cell.gene_counts[gene])
        data['score'].append(score)
        data['variance'].append(cell.gene_vars[gene])

    return pd.DataFrame(data)


def peripheral(cells, **kwargs):
    """
    Peripheral metric

    Fast mode: compute only score (skip theoretical variance).
    """
    processes = kwargs.get("processes", 2)
    compute_variance = kwargs.get("compute_variance", False)

    if compute_variance:
        cells = list(utils._iter_vars(cells, processes))
    else:
        for c in cells:
            if not getattr(c, "gene_vars", None):
                c.gene_vars = {g: np.nan for g in c.genes}

    if processes is None or processes <= 1:
        return pd.concat((_peripheral(c) for c in cells), ignore_index=True)

    with mp.Pool(processes=processes) as p:
        return pd.concat(p.imap_unordered(_peripheral, cells), ignore_index=True)


def _radial(cell, num_iterations, num_pairs, random_seed=None):
    """
    Helper radial function gets called by radial() for multiprocessing of each cell
    """
    data = {
        'metric':[],
        'cell_id':[],
        'annotation':[],
        'num_spots':[],
        'gene':[],
        'num_gene_spots':[],
        'score':[],
        'variance':[],
    }

    cell_centroid = np.mean(np.vstack(list(cell.boundaries.values())),axis=0)

    cell = cell.filter_genes_by_count(min_gene_spots=2)

    all_genes = np.array([g for z in cell.zslices for g in cell.spot_genes[z]])
    all_spots = np.array([xy for z in cell.zslices for xy in cell.spot_coords[z]], dtype=np.float64)

    var = (num_iterations+2)/(3*num_iterations)
    rng = _make_cell_rng(random_seed, "radial", str(cell.cell_id)) if random_seed is not None else None

    pre_calc_nulls = {}
    genes_iter = sorted(cell.gene_counts.items()) if rng is not None else cell.gene_counts.items()

    for gene,count in genes_iter:
        gene_spots = all_spots[all_genes == gene]
        if rng is not None:
            obs_dist = _mean_pairs_angle_rng(gene_spots, cell_centroid, num_pairs, rng)
        else:
            obs_dist = utils.random_mean_pairs_angle(gene_spots, cell_centroid, num_pairs)

        if count in pre_calc_nulls:
            null_dists = pre_calc_nulls[count]
        else:
            if rng is not None:
                null_dists = np.empty(num_iterations, dtype=np.float64)
                for i in range(num_iterations):
                    spot_inds = rng.choice(cell.n, count, replace=False)
                    null_dists[i] = _mean_pairs_angle_rng(
                        all_spots[spot_inds], cell_centroid, num_pairs, rng
                    )
            else:
                null_list = []
                for i in range(num_iterations):
                    spot_inds = np.random.choice(cell.n, count, replace=False)
                    null_list.append(
                        utils.random_mean_pairs_angle(all_spots[spot_inds], cell_centroid, num_pairs)
                    )
                null_dists = np.array(null_list)
            pre_calc_nulls[count] = null_dists

        tol = 1e-11 * max(1.0, abs(obs_dist)) if rng is not None else 0.0
        obs = int(np.sum(null_dists < (obs_dist - tol)))
        exp = num_iterations/2
        score = (exp-obs)/exp

        data['metric'].append('radial')
        data['cell_id'].append(cell.cell_id)
        data['annotation'].append(cell.annotation)
        data['num_spots'].append(cell.n)
        data['gene'].append(gene)
        data['num_gene_spots'].append(cell.gene_counts[gene])
        data['score'].append(score)
        data['variance'].append(var)

    return pd.DataFrame(data)


def radial(cells, **kwargs):
    """
    Radial metric
    """
    processes = kwargs.get('processes', 2)
    num_iterations = kwargs.get('num_iterations', 1000)
    num_pairs = kwargs.get('num_pairs', 4)
    random_seed = kwargs.get('random_seed', None)

    f = functools.partial(
        _radial,
        num_iterations=num_iterations,
        num_pairs=num_pairs,
        random_seed=random_seed,
    )

    if processes is None or processes <= 1:
        return pd.concat((f(c) for c in cells), ignore_index=True)

    with mp.Pool(processes=processes) as p:
        return pd.concat(p.imap_unordered(f, cells), ignore_index=True)


def _punctate(cell, num_iterations, num_pairs, random_seed=None):
    """
    Helper punctate function gets called by punctate() for multiprocessing of each cell
    """
    data = {
        'metric':[],
        'cell_id':[],
        'annotation':[],
        'num_spots':[],
        'gene':[],
        'num_gene_spots':[],
        'score':[],
        'variance':[],
    }

    cell = cell.filter_genes_by_count(min_gene_spots=2)

    all_genes = np.array([g for z in cell.zslices for g in cell.spot_genes[z]])
    all_spots = np.array([xy for z in cell.zslices for xy in cell.spot_coords[z]], dtype=np.float64)

    var = (num_iterations+2)/(3*num_iterations)
    rng = _make_cell_rng(random_seed, "punctate", str(cell.cell_id)) if random_seed is not None else None

    pre_calc_nulls = {}
    genes_iter = sorted(cell.gene_counts.items()) if rng is not None else cell.gene_counts.items()

    for gene,count in genes_iter:
        gene_spots = all_spots[all_genes == gene]
        if rng is not None:
            obs_dist = _mean_pairs_dist_rng(gene_spots, num_pairs, rng)
        else:
            obs_dist = utils.random_mean_pairs_dist(gene_spots, num_pairs)

        if count in pre_calc_nulls:
            null_dists = pre_calc_nulls[count]
        else:
            if rng is not None:
                null_dists = np.empty(num_iterations, dtype=np.float64)
                for i in range(num_iterations):
                    spot_inds = rng.choice(cell.n, count, replace=False)
                    null_dists[i] = _mean_pairs_dist_rng(
                        all_spots[spot_inds], num_pairs, rng
                    )
            else:
                null_list = []
                for i in range(num_iterations):
                    spot_inds = np.random.choice(cell.n, count, replace=False)
                    null_list.append(
                        utils.random_mean_pairs_dist(all_spots[spot_inds], num_pairs)
                    )
                null_dists = np.array(null_list)
            pre_calc_nulls[count] = null_dists

        tol = 1e-11 * max(1.0, abs(obs_dist)) if rng is not None else 0.0
        obs = int(np.sum(null_dists < (obs_dist - tol)))
        exp = len(null_dists)/2
        score = (exp-obs)/exp

        data['metric'].append('puncta')
        data['cell_id'].append(cell.cell_id)
        data['annotation'].append(cell.annotation)
        data['num_spots'].append(cell.n)
        data['gene'].append(gene)
        data['num_gene_spots'].append(cell.gene_counts[gene])
        data['score'].append(score)
        data['variance'].append(var)

    return pd.DataFrame(data)


def punctate(cells, **kwargs):
    """
    Punctate metric
    """
    processes = kwargs.get('processes', 2)
    num_iterations = kwargs.get('num_iterations', 1000)
    num_pairs = kwargs.get('num_pairs', 4)
    random_seed = kwargs.get('random_seed', None)

    f = functools.partial(
        _punctate,
        num_iterations=num_iterations,
        num_pairs=num_pairs,
        random_seed=random_seed,
    )

    if processes is None or processes <= 1:
        return pd.concat((f(c) for c in cells), ignore_index=True)

    with mp.Pool(processes=processes) as p:
        return pd.concat(p.imap_unordered(f, cells), ignore_index=True)


def _central(cell):
    """
    Helper central function gets called by central() for multiprocessing of each cell
    """
    data = {
        'metric':[],
        'cell_id':[],
        'annotation':[],
        'num_spots':[],
        'gene':[],
        'num_gene_spots':[],
        'score':[],
        'variance':[],
    }

    spot_genes = []
    spot_dists = []
    for zslice in cell.zslices:
        z_spot_coords = cell.spot_coords[zslice]
        z_spot_genes = cell.spot_genes[zslice]

        slice_centroid = np.mean(cell.boundaries[zslice], axis=0)
        dists = np.sum((z_spot_coords-slice_centroid)**2, axis=1)

        spot_genes.extend(z_spot_genes)
        spot_dists.extend(dists)

    spot_genes = np.array(spot_genes)
    spot_ranks = np.array(spot_dists).argsort().argsort()+1

    exp_med = (cell.n+1)/2
    for gene in cell.genes:
        gene_ranks = spot_ranks[spot_genes == gene]
        obs_med = np.median(gene_ranks)
        score = (exp_med-obs_med)/(exp_med-1)

        data['metric'].append('central')
        data['cell_id'].append(cell.cell_id)
        data['annotation'].append(cell.annotation)
        data['num_spots'].append(cell.n)
        data['gene'].append(gene)
        data['num_gene_spots'].append(cell.gene_counts[gene])
        data['score'].append(score)
        data['variance'].append(cell.gene_vars[gene])

    return pd.DataFrame(data)


def central(cells, **kwargs):
    """
    Central metric

    Fast mode: compute only score (skip theoretical variance).
    """
    processes = kwargs.get("processes", 2)
    compute_variance = kwargs.get("compute_variance", False)

    if compute_variance:
        cells = list(utils._iter_vars(cells, processes))
    else:
        for c in cells:
            if not getattr(c, "gene_vars", None):
                c.gene_vars = {g: np.nan for g in c.genes}

    if processes is None or processes <= 1:
        return pd.concat((_central(c) for c in cells), ignore_index=True)

    with mp.Pool(processes=processes) as p:
        return pd.concat(p.imap_unordered(_central, cells), ignore_index=True)
