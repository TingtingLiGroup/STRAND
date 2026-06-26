import collections
import numpy as np


class Cell:
    """
    Lightweight Cell object for SPRAWL metrics, built from transcripts + 2D boundary.

    This class is designed to satisfy what sprawl.metrics.py expects:
      - cell.cell_id, cell.annotation
      - cell.zslices
      - cell.boundaries[z] : (N,2) array-like
      - cell.spot_coords[z] : (M,2) ndarray
      - cell.spot_genes[z] : array-like of gene names
      - cell.n, cell.n_per_z
      - cell.genes, cell.gene_counts, cell.gene_vars
      - cell.filter_genes_by_count(...)
    """

    def __init__(
        self,
        cell_id: str,
        boundaries: dict,
        spot_coords: dict,
        spot_genes: dict,
        annotation: str = "NA",
    ):
        self.cell_id = str(cell_id)
        self.annotation = annotation

        # z-slices are keys of spot_coords / spot_genes / boundaries
        self.zslices = sorted(list(spot_coords.keys()))

        # required containers
        self.boundaries = boundaries            # dict[z] -> (N,2)
        self.spot_coords = spot_coords          # dict[z] -> ndarray(M,2)
        self.spot_genes = spot_genes            # dict[z] -> array/list[str]

        # counts per slice and total
        self.n_per_z = {z: int(len(self.spot_genes[z])) for z in self.zslices}
        self.n = int(sum(self.n_per_z.values()))

        # gene summary
        self.gene_counts = collections.Counter(
            g for z in self.zslices for g in self.spot_genes[z]
        )
        self.genes = sorted(list(self.gene_counts.keys()))

        # variance cache (filled by utils._iter_vars)
        self.gene_vars = {}

        # not strictly required for our metrics, but exists in original Cell
        self.ranked = False
        self.spot_ranks = {z: [] for z in self.zslices}
        self.spot_values = {z: [] for z in self.zslices}
        self.gene_med_ranks = {}

    def __repr__(self):
        return f"Cell-{self.cell_id}-{self.annotation}"

    def filter_genes_by_count(self, min_gene_spots=1, max_gene_spots=None):
        """
        Remove genes with gene-spot counts outside [min_gene_spots, max_gene_spots].
        This mirrors behavior used by SPRAWL radial/punctate metrics.

        Updates:
          - spot_genes/spot_coords per zslice
          - zslices (drops empty slices)
          - gene_counts/genes
          - n, n_per_z
          - clears gene_vars (must be recalculated)
        """
        # clear caches that depend on spots/genes
        self.gene_vars = {}
        self.spot_ranks = {}
        self.spot_values = {}
        self.gene_med_ranks = {}

        if not self.gene_counts:
            return self

        if max_gene_spots is None:
            max_gene_spots = max(self.gene_counts.values())

        # genes to delete
        del_genes = {
            g for g, c in self.gene_counts.items()
            if not (min_gene_spots <= c <= max_gene_spots)
        }

        # rebuild per zslice
        new_zs = []
        new_spot_genes = {}
        new_spot_coords = {}
        new_boundaries = {}

        new_n = 0
        for z in self.zslices:
            genes_z = self.spot_genes[z]
            coords_z = self.spot_coords[z]

            keep_genes = []
            keep_coords = []

            for g, xy in zip(genes_z, coords_z):
                if g in del_genes:
                    continue
                keep_genes.append(g)
                keep_coords.append(xy)

            if len(keep_genes) > 0:
                new_zs.append(z)
                new_spot_genes[z] = np.array(keep_genes, dtype=object)
                new_spot_coords[z] = np.asarray(keep_coords)
                new_boundaries[z] = self.boundaries[z]
                new_n += len(keep_genes)

        self.zslices = new_zs
        self.spot_genes = new_spot_genes
        self.spot_coords = new_spot_coords
        self.boundaries = new_boundaries

        self.n_per_z = {z: int(len(self.spot_genes[z])) for z in self.zslices}
        self.n = int(new_n)

        # recompute gene summary
        self.gene_counts = collections.Counter(
            g for z in self.zslices for g in self.spot_genes[z]
        )
        self.genes = sorted(list(self.gene_counts.keys()))

        # re-init caches
        self.spot_ranks = {z: [] for z in self.zslices}
        self.spot_values = {z: [] for z in self.zslices}

        return self