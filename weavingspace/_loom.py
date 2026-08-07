"""
MIT License

Copyright (c) 2021-26 David O'Sullivan & Luke Bergmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

"""
Handles combining the ordering of which strands are on top in weaves.

Initially in a biaxial weave intersection sites are encoded numerically:

  1 = warp is absent
  2 = weft is absent
  3 = both threads are absent
  4 = weft is on top
  5 = warp is on top

this tuple of dictionaries translates these to tuples listing the 'layer' order
depending on the input axis, where item 0 is the biaxial case with layers 0 and
1 only, item 1 is layers 0 and 1, item 2 is layers 1 and 2, and item 3 is
layers 2 and 0.
"""
_decode_orders:tuple[dict,dict,dict,dict] = (
  {1: (0,  ), 2: (1,  ), 3: (   ), 4: (0, 1), 5: (1, 0)}, # biaxial case
  {1: (1,  ), 2: (0,  ), 3: (   ), 4: (1, 0), 5: (0, 1)}, # triaxial 0|1
  {1: (2,  ), 2: (1,  ), 3: (   ), 4: (2, 1), 5: (1, 2)}, # triaxial 1|2
  {1: (0,  ), 2: (2,  ), 3: (   ), 4: (0, 2), 5: (2, 0)}, # triaxial 2|0
)

"""Nested dictionary to combine layer orders from 3 biaxial weaves
into a single consistent order of layers. Inconsistent combinations
will not return a value. Layers 0 and 1 are present in the first 'tier',
layers 1 and 2 in the second and 2 and 0 in the third.
"""
_combined_orderings:dict[tuple,dict] = {
  (    ): {(    ): {(    ): None,         # -- -- --  ->  All absent
                    (2,  ): (2,  )},      # -- -- 2-  ->  Only 2 present
           (1,  ): {(    ): (1,  )},      # -- 1- --  ->  Only 1 present
           (2,  ): {(2,  ): (2,  )}},     # -- 2- 2-  ->  Only 2 present
  (0,  ): {(    ): {(0,  ): (0,  ),       # 0- -- 0-  ->  Only 0 present
                    (    ): (0,  )},      # 0- -- --  ->  Only 0 present
           (2,  ): {(2, 0): (2, 0),       # 0- 2- 20  ->  2 > 0
                    (0, 2): (0, 2)}},     # 0- 2- 02  ->  0 > 2
  (1,  ): {(1,  ): {(    ): (1,  )},      # 1- 1- --  ->  Only 1 present
           (1, 2): {(2,  ): (1, 2)},      # 1- 12 2-  ->  1 > 2
           (2, 1): {(2,  ): (2, 1)}},     # 1- 21 2-  ->  2 > 1
  (0, 1): {(1,  ): {(0,  ): (0, 1)},      # 01 1- 0-  ->  0 > 1
           (1, 2): {(0, 2): (0, 1, 2)},   # 01 12 02  ->  0 > 1 > 2
           (2, 1): {(2, 0): (2, 0, 1),    # 01 21 20  ->  2 > 0 > 1
                    (0, 2): (0, 2, 1)}},  # 01 21 02  ->  0 > 2 > 1
  (1, 0): {(1,  ): {(0,  ): (1, 0)},      # 10 1- 0-  ->  1 > 0
           (1, 2): {(2, 0): (1, 2, 0),    # 10 12 20  ->  1 > 2 > 0
                    (0, 2): (1, 0, 2)},   # 10 12 02  ->  1 > 0 > 2
           (2, 1): {(2, 0): (2, 1, 0)}},  # 10 21 20  ->  2 > 1 > 0
}


@dataclass(slots=True)
class Loom:
  """A collection of weave matrices and associated attributes.

  Central to the loom object are the indices and orderings lists. These
  list the grid coordinate pairs and corresponding layer orderings at the
  site indexed by the coordinates. E.g., for a simple plain weave, we have:

    indices:   [(0, 0), (0, 1), (1, 0), (1, 1)]
    orderings: [(0, 1), (1, 0), (1, 0), (0, 1)]
  """

  indices:list[tuple]
  """grid coordinate pairs."""
  orderings:list
  """list of layer orders at site locations index by corresponding times in
  `indices`."""
  dimensions:tuple
  """maximum coordinate values in each direction."""
  orientations:tuple
  """angles (in degrees) which strands on each axis make with the x-axis.
  (0, -90) for 2 axes, (0, 120, 240) for 3."""
  n_axes:int
  """number of axes, 2 or 3."""

  def __init__(
      self,
      *matrices:np.ndarray,
    ) -> None:
    """Return a Loom instance.

    Takes either one or three weave matrices as input and initialises the loom
    based on these.
    """
    if len(matrices) == 1:
      m = matrices[0]
      self.dimensions = m.shape
      self.n_axes = len(self.dimensions)
      self.orientations = (0, -90)
      self.indices = [(i, j) for i in range(m.shape[0])
                             for j in range(m.shape[1])]
      self.orderings = [_decode_orders[0][m[ij]] for ij in self.indices]
    else:
      n_01, n_12, n_20 = [max(m.shape) for m in matrices]
      self.dimensions = (n_01, n_12, n_20)
      self.n_axes = len(self.dimensions)
      self.orientations = (0, 120, 240)
      all_indices = [(i, j, k) for i in range(n_01)
                               for j in range(n_12)
                               for k in range(n_20)]
      # parity is used to select coordinate triples in the grid see:
      # Nagy BN 2003. Shortest Paths in Triangular Grids with
      # Neighbourhood Sequences. Journal of Computing and Information
      # Technology 11 (2):111 https://doi.org/10.2498/cit.2003.02.04
      parity = (n_01 + n_12 + n_20 - 3) // 2
      self.indices = [x for x in all_indices if sum(x) in (parity, parity + 1)]
      m1, m2, m3 = matrices
      # extend the input 2D grids if needed
      m_01 = np.tile(m1, (np.lcm(n_01, m1.shape[0]) // m1.shape[0],
                          np.lcm(n_01, m1.shape[1]) // m1.shape[1]))
      m_12 = np.tile(m2, (np.lcm(n_12, m2.shape[0]) // m2.shape[0],
                          np.lcm(n_12, m2.shape[1]) // m2.shape[1]))
      m_20 = np.tile(m3, (np.lcm(n_20, m3.shape[0]) // m3.shape[0],
                          np.lcm(n_20, m3.shape[1]) // m3.shape[1]))
      # get the layer orders in each using 2 of the 3 coordinates
      order_01 = [_decode_orders[1][m_01[ij]]
                  for ij in [(x[1], x[0]) for x in self.indices]]
      order_12 = [_decode_orders[2][m_12[ij]]
                  for ij in [(x[2], x[1]) for x in self.indices]]
      order_20 = [_decode_orders[3][m_20[ij]]
                  for ij in [(x[0], x[2]) for x in self.indices]]
      # # combine orders from the three matrices stacked
      self.orderings = [
        self._combine_orders(abc)
        for abc in zip(order_01, order_12, order_20, strict = True)]


  # convenience wrapper for the combined_orderings dictionary
  # missing values return "NA"
  def _combine_orders(
      self,
      orders:tuple[tuple[int],...],
    ) -> tuple[int]|str:
    try:
      result = _combined_orderings[orders[0]][orders[1]][orders[2]]
    except KeyError as e:
      e.add_note(f"Unable to determine unique ordering on {orders}")
      return "NA"
    else:
      return result
