'''
PM4Py – A Process Mining Library for Python
Copyright (C) 2026 Process Intelligence Solutions GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see this software project's root or
visit <https://www.gnu.org/licenses/>.

Website: https://processintelligence.solutions
Contact: info@processintelligence.solutions
'''


def tree_simplicity(node, inflection_point=75, steepness=2) -> float:
    """
    Calculates a structural simplicity percentage for a ProcessTree.
    Returns a float between 0.0 (unreadable spaghetti) and 1.0 (perfectly simple).

    This metric aggregates structural penalties (nodes, depth, width, and routing elements)
    and normalizes them using a Hill Equation (S-curve) to mimic human comprehension.
    It provides a "grace period" where moderate trees remain highly scored, while
    exponentially penalizing massive, tangled models.

    Penalty Weights:
      - +1.0 per node (base size)
      - +1.0 per level of maximum depth (nesting)
      - +1.0 per maximum width beyond 1 (heavy branching)
      - +2.0 per Tau (silent transition / XOR skip)
      - +2.0 per explicit Skip annotation
      - +0.5 per Start/Stop annotation

    Parameters:
      - node: The root ProcessTree or EnhancedProcessTree node.
      - inflection_point: The penalty threshold where the score drops to exactly 50% (0.50).
                          Increasing this stretches the "grace period" for moderate trees.
      - steepness: The rate of decay after the inflection point. A value of 2.0 ensures a
                   gradual drop so that standard "spaghetti" models sit around 30-40%
                   rather than crashing immediately to zero.
    """

    def analyze_tree(n, current_depth=1):
        nodes = 1

        taus = 1 if (len(n.children) == 0 and n.label is None) else 0
        starts = 1 if getattr(n, 'start', False) else 0
        stops = 1 if getattr(n, 'stop', False) else 0
        skips = 1 if getattr(n, 'skip', False) else 0

        max_depth = current_depth
        max_width = len(n.children)

        for child in n.children:
            c_nodes, c_taus, c_starts, c_stops, c_skips, c_depth, c_width = analyze_tree(child, current_depth + 1)

            nodes += c_nodes
            taus += c_taus
            starts += c_starts
            stops += c_stops
            skips += c_skips

            max_depth = max(max_depth, c_depth)
            max_width = max(max_width, c_width)

        return nodes, taus, starts, stops, skips, max_depth, max_width

    nodes, taus, starts, stops, skips, max_depth, max_width = analyze_tree(node)

    width_penalty = max(0, max_width - 1)

    penalty = (
            (nodes - 1) * 1.0 +
            (max_depth - 1) * 1.0 +
            (width_penalty) * 1.0 +
            (taus * 1.0) +
            (skips * 1.0) +
            (starts * 0.5) +
            (stops * 0.5)
    )

    if penalty == 0:
        return 1.0

    simplicity_score = 1.0 / (1.0 + (penalty / inflection_point) ** steepness)

    return simplicity_score