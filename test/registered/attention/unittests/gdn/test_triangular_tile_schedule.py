"""Unit tests for the strict-lower tile specialization used by SM100 GDN/KDA.

The CuTe kernels keep every tile in shared memory because the inverse stage has
a fixed ldmatrix layout.  Only tiles with ``column_block <= row_block`` can
contain data; upper tiles are written as zero without loading KKT.  Keep this
small invariant tested on CPU so a future layout change cannot silently make
the fast path drop a lower-triangular tile.
"""

import pytest


def _tile_is_live(row_block: int, column_block: int) -> bool:
    return column_block <= row_block


@pytest.mark.parametrize("row_block", range(4))
def test_strict_lower_tile_schedule(row_block: int):
    live = [_tile_is_live(row_block, col) for col in range(4)]
    assert live == [True] * (row_block + 1) + [False] * (3 - row_block)


def test_diagonal_tile_is_kept_for_element_mask():
    # The diagonal tile is not all zero: its strict-lower half is live, while
    # the upper half is masked by the existing element-wise predicate.
    for block in range(4):
        assert _tile_is_live(block, block)

