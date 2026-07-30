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
import copy
from pm4py.objects.enhanced_process_tree.obj import EnhancedProcessTree


def convert_to_enhanced_process_tree(node, annotations=None, parent=None):
    """
    Recursively deep-clones a standard ProcessTree into an EnhancedProcessTree,
    injecting the start, stop, and skip annotations onto the leaf nodes.

    :param node: The current ProcessTree node to clone.
    :param annotations: Dictionary of annotations {'activity': ['stop', 'skip']}.
    :param parent: The EnhancedProcessTree parent node (used for recursion).
    :return: The root of the newly created EnhancedProcessTree.
    """
    if annotations is None:
        annotations = {}

    enhanced_node = EnhancedProcessTree(
        operator=node.operator,
        parent=parent,
        label=node.label
    )

    if node.operator is None and node.label is not None:
        if node.label in annotations:
            flags = annotations[node.label]
            enhanced_node.start = "start" in flags
            enhanced_node.stop = "stop" in flags
            enhanced_node.skip = "skip" in flags

    for child in node.children:
        enhanced_child = convert_to_enhanced_process_tree(child, annotations, parent=enhanced_node)
        enhanced_node.children.append(enhanced_child)

    return enhanced_node


def is_enhanced(tree):
    """
    Checks whether the given tree and all of its descendants are
    enhanced process trees.
    """
    if not isinstance(tree, EnhancedProcessTree):
        return False
    return all(is_enhanced(c) for c in tree.children)


def get_annotated_nodes(tree):
    """
    Returns all nodes carrying at least one annotation.
    """
    result = [tree] if (tree.start or tree.stop or tree.skip) else []
    for child in tree.children:
        result.extend(get_annotated_nodes(child))
    return result