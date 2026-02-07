
import itertools
import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)


def aggregate_hierarchy(
    conn: sqlite3.Connection,
    base_tile_size_y: int,
    base_tile_size_x: int,
    num_levels: int = 3,
) -> dict:
    """
    Build hierarchy levels 1..num_levels by aggregating 2×2 tile groups.

    Each level L groups level-(L-1) tiles into 2×2 blocks, sums raw
    voxel counts, and recomputes enrichment from the pooled counts.

    Args:
        conn: Open sqlite3 connection to the temp database
        base_tile_size_y: Level-0 tile height in voxels
        base_tile_size_x: Level-0 tile width in voxels
        num_levels: Number of additional hierarchy levels (default 3)

    Returns:
        Dict with per-level row counts for logging
    """
    if num_levels < 1:
        return {}

    summary = {}

    for level in range(1, num_levels + 1):
        # At level L the grouping factor doubles each step relative to L-1
        group_y = base_tile_size_y * (2 ** level)
        group_x = base_tile_size_x * (2 ** level)
        source = level - 1

        logger.info(
            f"Hierarchy level {level}: grouping level-{source} tiles "
            f"into {group_y}×{group_x} blocks..."
        )

        n_cs = _aggregate_channel_stats(conn, source, level, group_y, group_x)
        n_pw = _aggregate_pairwise_metrics(conn, source, level, group_y, group_x)
        n_co = _aggregate_combinations(conn, source, level, group_y, group_x)

        conn.commit()

        summary[level] = {
            "channel_stats": n_cs,
            "pairwise_metrics": n_pw,
            "combination_tiles": n_co,
        }
        logger.info(
            f"  -> {n_cs} channel_stats, {n_pw} pairwise_metrics, "
            f"{n_co} combination tile rows"
        )

    return summary


# =====================================================================
# channel_stats aggregation
# =====================================================================

def _aggregate_channel_stats(
    conn: sqlite3.Connection,
    source_level: int,
    target_level: int,
    group_y: int,
    group_x: int,
) -> int:
    """
    Pool channel_stats from source_level -> target_level.

    Per coarsened-tile-group x channel:
      active_voxels   = SUM
      total_voxels    = SUM
      active_fraction = SUM(active) / SUM(total)   [recomputed]
      mean_intensity  = weighted avg by total_voxels
      max_intensity   = MAX
      threshold_value = AVG (informational only)
    """
    sql = """
        INSERT INTO channel_stats (
            tile_x0, tile_x1, tile_y0, tile_y1,
            hierarchy_level, channel, channel_idx,
            threshold_value, active_voxels, active_fraction,
            mean_intensity, max_intensity, total_voxels
        )
        SELECT
            (tile_x0 / :gx) * :gx,
            MAX(tile_x1),
            (tile_y0 / :gy) * :gy,
            MAX(tile_y1),

            :target,
            channel,
            channel_idx,

            AVG(threshold_value),
            SUM(active_voxels),
            CASE WHEN SUM(total_voxels) > 0
                 THEN CAST(SUM(active_voxels) AS REAL) / SUM(total_voxels)
                 ELSE 0.0 END,
            CASE WHEN SUM(total_voxels) > 0
                 THEN SUM(mean_intensity * total_voxels) / SUM(total_voxels)
                 ELSE NULL END,
            MAX(max_intensity),
            SUM(total_voxels)

        FROM channel_stats
        WHERE hierarchy_level = :source
        GROUP BY (tile_x0 / :gx), (tile_y0 / :gy), channel_idx
    """
    cur = conn.execute(sql, {
        "gx": group_x, "gy": group_y,
        "source": source_level, "target": target_level,
    })
    return cur.rowcount


# =====================================================================
# pairwise_metrics aggregation
# =====================================================================

def _aggregate_pairwise_metrics(
    conn: sqlite3.Connection,
    source_level: int,
    target_level: int,
    group_y: int,
    group_x: int,
) -> int:
    """
    Pool pairwise_metrics from source_level -> target_level.

    Per coarsened-tile-group x dilation x channel pair:
      count_a, count_b, count_ab = SUM
      total_voxels               = SUM  (joined from channel_stats)

    Then recompute all derived metrics from pooled counts.
    """
    sql = """
        INSERT INTO pairwise_metrics (
            tile_x0, tile_x1, tile_y0, tile_y1,
            hierarchy_level, dilation,
            channel_a, channel_b,
            channel_a_idx, channel_b_idx,
            count_a, count_b, count_ab,
            prob_b_given_a, prob_a_given_b,
            jaccard, overlap_coeff, enrichment
        )
        SELECT
            new_x0, new_x1, new_y0, new_y1,
            :target,
            dilation,
            channel_a, channel_b,
            channel_a_idx, channel_b_idx,
            sa, sb, sab,

            -- P(B|A)
            CASE WHEN sa > 0  THEN CAST(sab AS REAL) / sa  ELSE NULL END,
            -- P(A|B)
            CASE WHEN sb > 0  THEN CAST(sab AS REAL) / sb  ELSE NULL END,
            -- Jaccard
            CASE WHEN (sa + sb - sab) > 0
                 THEN CAST(sab AS REAL) / (sa + sb - sab)   ELSE NULL END,
            -- Overlap coefficient
            CASE WHEN MIN(sa, sb) > 0
                 THEN CAST(sab AS REAL) / MIN(sa, sb)       ELSE NULL END,
            -- Enrichment  E = |A int B| * total / (|A| * |B|)
            CASE WHEN sa > 0 AND sb > 0 AND stot > 0
                 THEN (CAST(sab AS REAL) * stot) / (CAST(sa AS REAL) * sb)
                 ELSE NULL END

        FROM (
            SELECT
                (p.tile_x0 / :gx) * :gx  AS new_x0,
                MAX(p.tile_x1)            AS new_x1,
                (p.tile_y0 / :gy) * :gy  AS new_y0,
                MAX(p.tile_y1)            AS new_y1,
                p.dilation,
                p.channel_a,  p.channel_b,
                p.channel_a_idx, p.channel_b_idx,
                SUM(p.count_a)   AS sa,
                SUM(p.count_b)   AS sb,
                SUM(p.count_ab)  AS sab,
                SUM(tv.tvox)     AS stot
            FROM pairwise_metrics p
            LEFT JOIN (
                -- one total_voxels per source-level tile
                SELECT tile_x0, tile_y0,
                       MAX(total_voxels) AS tvox
                FROM channel_stats
                WHERE hierarchy_level = :source
                GROUP BY tile_x0, tile_y0
            ) tv ON p.tile_x0 = tv.tile_x0
                 AND p.tile_y0 = tv.tile_y0
            WHERE p.hierarchy_level = :source
            GROUP BY (p.tile_x0 / :gx), (p.tile_y0 / :gy),
                     p.dilation, p.channel_a_idx, p.channel_b_idx
        )
    """
    cur = conn.execute(sql, {
        "gx": group_x, "gy": group_y,
        "source": source_level, "target": target_level,
    })
    return cur.rowcount


# =====================================================================
# combinations + tiles aggregation
# =====================================================================

def _aggregate_combinations(
    conn: sqlite3.Connection,
    source_level: int,
    target_level: int,
    group_y: int,
    group_x: int,
) -> int:
    """
    Pool combinations and their tile rows from source -> target level.

    1. For each unique (channels, dilation) at source_level, create a
       new combination at target_level with summed overlap_count.
    2. Group the tile rows into coarsened tiles with summed counts.
    3. Recompute enrichment from pooled global channel counts.
    """
    # Get distinct combination signatures at source level
    source_combos = conn.execute("""
        SELECT DISTINCT channels, channel_indices, channel_count, dilation
        FROM combinations
        WHERE hierarchy_level = ?
    """, (source_level,)).fetchall()

    total_tile_rows = 0

    for channels, channel_indices, channel_count, dilation in source_combos:
        # Collect source combination ids for this signature
        src_ids = [r[0] for r in conn.execute("""
            SELECT id FROM combinations
            WHERE channels = ? AND dilation = ? AND hierarchy_level = ?
        """, (channels, dilation, source_level)).fetchall()]

        if not src_ids:
            continue

        # Sum overlap count across all source combos for this signature
        ph = ",".join("?" * len(src_ids))
        total_overlap = conn.execute(
            f"SELECT SUM(overlap_count) FROM combinations WHERE id IN ({ph})",
            src_ids,
        ).fetchone()[0] or 0

        # Insert the new combination row at target level
        cur = conn.execute("""
            INSERT INTO combinations
                (channels, channel_indices, channel_count, dilation,
                 hierarchy_level, overlap_count, total_voxels,
                 enrichment_ratio, higher_order_enrichment)
            VALUES (?, ?, ?, ?, ?, ?, 0, NULL, NULL)
        """, (channels, channel_indices, channel_count, dilation,
              target_level, total_overlap))
        new_combo_id = cur.lastrowid

        # Insert grouped tile rows
        cur2 = conn.execute(f"""
            INSERT INTO tiles (
                combination_id,
                tile_x0, tile_x1, tile_y0, tile_y1,
                count, enrichment_ratio
            )
            SELECT
                ?,
                (tile_x0 / ?) * ?,
                MAX(tile_x1),
                (tile_y0 / ?) * ?,
                MAX(tile_y1),
                SUM(count),
                NULL
            FROM tiles
            WHERE combination_id IN ({ph})
            GROUP BY (tile_x0 / ?), (tile_y0 / ?)
        """, (new_combo_id,
              group_x, group_x, group_y, group_y,
              *src_ids,
              group_x, group_y))
        total_tile_rows += cur2.rowcount

    # Recompute enrichment from pooled counts at this level
    _recompute_enrichment(conn, target_level)

    return total_tile_rows


# =====================================================================
# Enrichment recomputation from pooled counts
# =====================================================================

def _recompute_enrichment(
    conn: sqlite3.Connection,
    hierarchy_level: int,
) -> None:
    """
    Recompute enrichment_ratio (and higher_order_enrichment) for all
    combinations at a given hierarchy level, using the pooled counts
    in channel_stats and pairwise_metrics at that level.
    """
    # Global channel counts at this level (summed across all tiles)
    rows = conn.execute("""
        SELECT channel_idx,
               SUM(active_voxels)  AS total_active,
               SUM(total_voxels)   AS total_vox
        FROM channel_stats
        WHERE hierarchy_level = ?
        GROUP BY channel_idx
    """, (hierarchy_level,)).fetchall()

    ch_active = {r[0]: r[1] for r in rows}
    global_total = max((r[2] for r in rows), default=0) if rows else 0

    if global_total == 0:
        return

    # --- Pairwise combinations ---
    pairs = conn.execute("""
        SELECT id, channel_indices, overlap_count
        FROM combinations
        WHERE hierarchy_level = ? AND channel_count = 2
    """, (hierarchy_level,)).fetchall()

    for combo_id, idx_str, overlap in pairs:
        indices = [int(x) for x in idx_str.split("|")]
        ca = ch_active.get(indices[0], 0)
        cb = ch_active.get(indices[1], 0)

        e = (overlap * global_total) / (ca * cb) if ca > 0 and cb > 0 else None

        conn.execute(
            "UPDATE combinations SET enrichment_ratio = ?, total_voxels = ? WHERE id = ?",
            (e, global_total, combo_id),
        )

    # --- Higher-order combinations ---
    higher = conn.execute("""
        SELECT id, channel_indices, channel_count, overlap_count
        FROM combinations
        WHERE hierarchy_level = ? AND channel_count > 2
    """, (hierarchy_level,)).fetchall()

    for combo_id, idx_str, k, overlap in higher:
        indices = [int(x) for x in idx_str.split("|")]
        counts = [ch_active.get(i, 0) for i in indices]

        # E_indep = overlap * total^(k-1) / product(counts)
        e_indep = None
        if all(c > 0 for c in counts):
            product = 1.0
            for c in counts:
                product *= c
            e_indep = (overlap * (global_total ** (k - 1))) / product

        # E_higher = overlap * product(counts) / product(pairwise_overlaps)
        e_higher = None
        pw_product = 1.0
        pw_ok = True
        for i, j in itertools.combinations(indices, 2):
            a, b = min(i, j), max(i, j)
            row = conn.execute("""
                SELECT SUM(count_ab) FROM pairwise_metrics
                WHERE hierarchy_level = ?
                  AND channel_a_idx = ? AND channel_b_idx = ?
            """, (hierarchy_level, a, b)).fetchone()
            pw = row[0] if row and row[0] else 0
            if pw == 0:
                pw_ok = False
                break
            pw_product *= pw

        if pw_ok and pw_product > 0 and all(c > 0 for c in counts):
            count_product = 1.0
            for c in counts:
                count_product *= c
            e_higher = (overlap * count_product) / pw_product

        conn.execute(
            "UPDATE combinations SET enrichment_ratio = ?, higher_order_enrichment = ?, total_voxels = ? WHERE id = ?",
            (e_indep, e_higher, global_total, combo_id),
        )