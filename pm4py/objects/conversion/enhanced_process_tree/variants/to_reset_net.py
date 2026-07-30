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
import time
from collections import namedtuple
from enum import Enum

from pm4py.objects.enhanced_process_tree.obj import EnhancedProcessTree
from pm4py.objects.enhanced_process_tree.utils.generic import is_enhanced
from pm4py.objects.petri_net.obj import Marking, PetriNet, ResetNet
from pm4py.objects.petri_net.utils import reduction
from pm4py.objects.petri_net.utils.petri_utils import (
    add_arc_from_to,
    remove_place,
)
from pm4py.objects.process_tree.obj import Operator
from pm4py.util import exec_utils

from pm4py.objects.conversion.process_tree.variants.to_petri_net import (
    Counts,
    check_tau_mandatory_at_final_marking,
    check_tau_mandatory_at_initial_marking,
    get_new_hidden_trans,
    get_new_place,
    get_transition,
)


class Parameters(Enum):
    APPLY_REDUCTION = "apply_reduction"
    REMOVE_DANGLING_PLACES = "remove_dangling_places"


# A bypass is a silent transition that leaves the normal control flow of a
# subtree: either straight to the global sink (a 'stop' annotation) or to the
# final place of the parent (a 'skip' annotation). They are tracked explicitly
# instead of being recovered from transition names later on.
Bypass = namedtuple("Bypass", ["transition", "kind", "target"])


def add_reset_arc_from_to(fr, to, net, weight=1):
    """
    Adds a reset arc between a place and a transition.

    Parameters
    ------------
    fr
        Source place
    to
        Target transition
    net
        Reset net
    weight
        Weight of the arc

    Returns
    ------------
    arc
        The newly created reset arc
    """
    arc = ResetNet.ResetArc(fr, to, weight)
    net.arcs.add(arc)
    fr.out_arcs.add(arc)
    to.in_arcs.add(arc)
    return arc


def contains_start(tree):
    """
    Recursively checks whether a subtree contains a node annotated as a start point.

    Parameters
    ------------
    tree
        Enhanced process tree

    Returns
    ------------
    boolean
        True if the subtree contains a start annotation
    """
    if tree.start:
        return True
    for child in tree.children:
        if contains_start(child):
            return True
    return False


def recursively_add_tree(
    parent_tree,
    tree,
    net,
    initial_entity_subtree,
    final_entity_subtree,
    counts,
    rec_depth,
    force_add_skip=False,
    global_sink=None,
    parent_final_place=None,
    global_source=None,
    bypasses=None,
):
    """
    Recursively adds the subtrees to the reset net

    Parameters
    -----------
    parent_tree
        Parent tree
    tree
        Current subtree
    net
        Reset net
    initial_entity_subtree
        Initial entity (place/transition) that should be attached from the subtree
    final_entity_subtree
        Final entity (place/transition) that should be attached from the subtree
    counts
        Counts object (keeps the number of places, transitions and hidden transitions)
    rec_depth
        Recursion depth of the current iteration
    force_add_skip
        Boolean value that tells if the addition of a skip is mandatory
    global_sink
        The end place of the whole reset net
    parent_final_place
        The end place of the parent
    global_source
        The start place of the whole reset net
    bypasses
        Accumulator collecting every bypass transition created during the recursion

    Returns
    ----------
    net
        Updated reset net
    counts
        Updated counts object
    final_place
        Last place added in this recursion
    """
    if bypasses is None:
        bypasses = []

    if type(initial_entity_subtree) is PetriNet.Transition:
        initial_place = get_new_place(counts)
        net.places.add(initial_place)
        add_arc_from_to(initial_entity_subtree, initial_place, net)
    else:
        initial_place = initial_entity_subtree

    if tree.start and global_source is not None and initial_place != global_source:
        tau_start = get_new_hidden_trans(counts, type_trans="tau_start")
        net.transitions.add(tau_start)

        # Connect: global source -> tau_start -> local initial place
        add_arc_from_to(global_source, tau_start, net)
        add_arc_from_to(tau_start, initial_place, net)

    if (
        final_entity_subtree is not None
        and type(final_entity_subtree) is PetriNet.Place
    ):
        final_place = final_entity_subtree
    else:
        final_place = get_new_place(counts)
        net.places.add(final_place)
        if (
            final_entity_subtree is not None
            and type(final_entity_subtree) is PetriNet.Transition
        ):
            add_arc_from_to(final_place, final_entity_subtree, net)

    tree_childs = [child for child in tree.children]

    if force_add_skip:
        invisible = get_new_hidden_trans(counts, type_trans="skip")
        add_arc_from_to(initial_place, invisible, net)
        add_arc_from_to(invisible, final_place, net)

    if tree.operator is None:
        trans = tree
        if trans.label is None:
            petri_trans = get_new_hidden_trans(counts, type_trans="skip")
        else:
            petri_trans = get_transition(counts, trans.label)
        net.transitions.add(petri_trans)
        add_arc_from_to(initial_place, petri_trans, net)
        add_arc_from_to(petri_trans, final_place, net)

    if tree.operator == Operator.XOR:
        for subtree in tree_childs:
            if subtree.start and global_source is not None:
                # Isolate the branch entry so the start bypass cannot be taken
                # by the other branches of the choice.
                isolated_start_place = get_new_place(counts)
                net.places.add(isolated_start_place)

                tau_enter = get_new_hidden_trans(
                    counts, type_trans="tau_xor_enter"
                )
                net.transitions.add(tau_enter)
                add_arc_from_to(initial_place, tau_enter, net)
                add_arc_from_to(tau_enter, isolated_start_place, net)

                target_initial_place = isolated_start_place
            else:
                target_initial_place = initial_place

            if subtree.stop and global_sink is not None:
                # Isolate the branch exit, then rejoin the shared XOR end place
                # through a silent transition.
                isolated_place = get_new_place(counts)
                net.places.add(isolated_place)

                tau_continue = get_new_hidden_trans(
                    counts, type_trans="tau_xor_continue"
                )
                net.transitions.add(tau_continue)
                add_arc_from_to(isolated_place, tau_continue, net)
                add_arc_from_to(tau_continue, final_place, net)

                target_final_place = isolated_place
            else:
                target_final_place = final_place

            net, counts, intermediate_place = recursively_add_tree(
                tree,
                subtree,
                net,
                target_initial_place,
                target_final_place,
                counts,
                rec_depth + 1,
                global_sink=global_sink,
                parent_final_place=final_place,
                global_source=global_source,
                bypasses=bypasses,
            )

    elif tree.operator == Operator.OR:
        new_initial_trans = get_new_hidden_trans(counts, type_trans="tauSplit")
        net.transitions.add(new_initial_trans)
        add_arc_from_to(initial_place, new_initial_trans, net)
        new_final_trans = get_new_hidden_trans(counts, type_trans="tauJoin")
        net.transitions.add(new_final_trans)
        add_arc_from_to(new_final_trans, final_place, net)
        terminal_place = get_new_place(counts)
        net.places.add(terminal_place)
        add_arc_from_to(terminal_place, new_final_trans, net)
        first_place = get_new_place(counts)
        net.places.add(first_place)
        add_arc_from_to(new_initial_trans, first_place, net)

        for subtree in tree_childs:
            subtree_init_place = get_new_place(counts)
            net.places.add(subtree_init_place)
            add_arc_from_to(new_initial_trans, subtree_init_place, net)
            subtree_start_place = get_new_place(counts)
            net.places.add(subtree_start_place)
            subtree_end_place = get_new_place(counts)
            net.places.add(subtree_end_place)
            trans_start = get_new_hidden_trans(
                counts, type_trans="inclusiveStart"
            )
            trans_later = get_new_hidden_trans(
                counts, type_trans="inclusiveLater"
            )
            trans_skip = get_new_hidden_trans(
                counts, type_trans="inclusiveSkip"
            )
            net.transitions.add(trans_start)
            net.transitions.add(trans_later)
            net.transitions.add(trans_skip)
            add_arc_from_to(first_place, trans_start, net)
            add_arc_from_to(subtree_init_place, trans_start, net)
            add_arc_from_to(trans_start, subtree_start_place, net)
            add_arc_from_to(trans_start, terminal_place, net)

            add_arc_from_to(terminal_place, trans_later, net)
            add_arc_from_to(subtree_init_place, trans_later, net)
            add_arc_from_to(trans_later, subtree_start_place, net)
            add_arc_from_to(trans_later, terminal_place, net)

            add_arc_from_to(terminal_place, trans_skip, net)
            add_arc_from_to(subtree_init_place, trans_skip, net)
            add_arc_from_to(trans_skip, terminal_place, net)
            add_arc_from_to(trans_skip, subtree_end_place, net)

            add_arc_from_to(subtree_end_place, new_final_trans, net)

            net, counts, intermediate_place = recursively_add_tree(
                tree,
                subtree,
                net,
                subtree_start_place,
                subtree_end_place,
                counts,
                rec_depth + 1,
                global_sink=global_sink,
                parent_final_place=final_place,
                global_source=global_source,
                bypasses=bypasses,
            )

    elif tree.operator == Operator.PARALLEL:
        new_initial_trans = get_new_hidden_trans(counts, type_trans="tauSplit")
        net.transitions.add(new_initial_trans)
        add_arc_from_to(initial_place, new_initial_trans, net)
        new_final_trans = get_new_hidden_trans(counts, type_trans="tauJoin")
        net.transitions.add(new_final_trans)
        add_arc_from_to(new_final_trans, final_place, net)

        places_before = set(net.places)
        bypasses_before = len(bypasses)

        has_internal_start = any(contains_start(child) for child in tree_childs)

        if has_internal_start and global_source is not None:
            # The AND block intercepts the start signal, so that starting
            # inside one branch still enables all the other branches.
            tau_start_split = get_new_hidden_trans(
                counts, type_trans="tau_start_split"
            )
            net.transitions.add(tau_start_split)
            add_arc_from_to(global_source, tau_start_split, net)

        for subtree in tree_childs:
            # Explicitly create the normal starting place for this branch
            subtree_init_place = get_new_place(counts)
            net.places.add(subtree_init_place)
            add_arc_from_to(new_initial_trans, subtree_init_place, net)

            branch_global_source = global_source

            if has_internal_start and global_source is not None:
                if contains_start(subtree):
                    # This branch holds the start node: pass a synced source down
                    branch_sync_source = get_new_place(counts)
                    net.places.add(branch_sync_source)
                    add_arc_from_to(tau_start_split, branch_sync_source, net)
                    branch_global_source = branch_sync_source
                else:
                    # This branch does not hold the start node: it starts normally.
                    add_arc_from_to(tau_start_split, subtree_init_place, net)
                    # Sever the source so it does not build a false bypass inside itself
                    branch_global_source = None

            net, counts, intermediate_place = recursively_add_tree(
                tree,
                subtree,
                net,
                subtree_init_place,
                new_final_trans,
                counts,
                rec_depth + 1,
                global_sink=global_sink,
                parent_final_place=final_place,
                global_source=branch_global_source,
                bypasses=bypasses,
            )

        and_places = set(net.places) - places_before
        new_bypasses = bypasses[bypasses_before:]

        # Leaving an AND block through a bypass must clear the tokens that are
        # still sitting in the concurrent branches, otherwise the net would keep
        # firing them after the process has already left the block.
        for bypass in new_bypasses:
            if bypass.kind == "stop":
                is_valid_bypass = True
            elif bypass.kind == "skip":
                is_valid_bypass = bypass.target is final_place
            else:
                is_valid_bypass = False

            if not is_valid_bypass:
                continue

            regular_inputs = {arc.source for arc in bypass.transition.in_arcs}
            for p in and_places:
                # No reset arc if the place is already a regular input of the bypass
                if p not in regular_inputs:
                    add_reset_arc_from_to(p, bypass.transition, net)

    elif tree.operator == Operator.INTERLEAVING:
        new_initial_trans = get_new_hidden_trans(counts, type_trans="tauSplit")
        net.transitions.add(new_initial_trans)
        add_arc_from_to(initial_place, new_initial_trans, net)
        new_final_trans = get_new_hidden_trans(counts, type_trans="tauJoin")
        net.transitions.add(new_final_trans)
        add_arc_from_to(new_final_trans, final_place, net)

        control_place = get_new_place(counts)
        net.places.add(control_place)

        add_arc_from_to(new_initial_trans, control_place, net)
        add_arc_from_to(control_place, new_final_trans, net)

        for subtree in tree_childs:
            placeI = get_new_place(counts)
            net.places.add(placeI)
            iTrans = get_new_hidden_trans(counts, type_trans="iTrans")
            net.transitions.add(iTrans)
            placeF = get_new_place(counts)
            net.places.add(placeF)
            fTrans = get_new_hidden_trans(counts, type_trans="fTrans")
            net.transitions.add(fTrans)

            add_arc_from_to(new_initial_trans, placeI, net)
            add_arc_from_to(placeI, iTrans, net)
            add_arc_from_to(fTrans, placeF, net)
            add_arc_from_to(placeF, new_final_trans, net)

            add_arc_from_to(control_place, iTrans, net)
            add_arc_from_to(fTrans, control_place, net)

            net, counts, intermediate_place = recursively_add_tree(
                tree,
                subtree,
                net,
                iTrans,
                fTrans,
                counts,
                rec_depth + 1,
                global_sink=global_sink,
                parent_final_place=final_place,
                global_source=global_source,
                bypasses=bypasses,
            )

    elif tree.operator == Operator.SEQUENCE:
        intermediate_place = initial_place
        for i in range(len(tree_childs)):
            final_connection_place = None
            if i == len(tree_childs) - 1:
                final_connection_place = final_place
            net, counts, intermediate_place = recursively_add_tree(
                tree,
                tree_childs[i],
                net,
                intermediate_place,
                final_connection_place,
                counts,
                rec_depth + 1,
                global_sink=global_sink,
                parent_final_place=final_place,
                global_source=global_source,
                bypasses=bypasses,
            )

    elif tree.operator == Operator.LOOP:
        new_initial_place = get_new_place(counts)
        net.places.add(new_initial_place)
        init_loop_trans = get_new_hidden_trans(counts, type_trans="init_loop")
        net.transitions.add(init_loop_trans)
        add_arc_from_to(initial_place, init_loop_trans, net)
        add_arc_from_to(init_loop_trans, new_initial_place, net)
        initial_place = new_initial_place
        loop_trans = get_new_hidden_trans(counts, type_trans="loop")
        net.transitions.add(loop_trans)
        if len(tree_childs) == 1:
            net, counts, intermediate_place = recursively_add_tree(
                tree,
                tree_childs[0],
                net,
                initial_place,
                final_place,
                counts,
                rec_depth + 1,
                global_sink=global_sink,
                parent_final_place=final_place,
                global_source=global_source,
                bypasses=bypasses,
            )
            add_arc_from_to(final_place, loop_trans, net)
            add_arc_from_to(loop_trans, initial_place, net)
        else:
            net, counts, int1 = recursively_add_tree(
                tree,
                tree_childs[0],
                net,
                initial_place,
                None,
                counts,
                rec_depth + 1,
                global_sink=global_sink,
                parent_final_place=final_place,
                global_source=global_source,
                bypasses=bypasses,
            )
            int2 = None
            for i in range(1, len(tree_childs)):
                child = tree_childs[i]
                if child.start and global_source is not None:
                    isolated_loop_start = get_new_place(counts)
                    net.places.add(isolated_loop_start)

                    tau_loop_enter = get_new_hidden_trans(
                        counts, type_trans="tau_loop_enter"
                    )
                    net.transitions.add(tau_loop_enter)
                    add_arc_from_to(int1, tau_loop_enter, net)
                    add_arc_from_to(tau_loop_enter, isolated_loop_start, net)

                    target_initial_place = isolated_loop_start
                else:
                    target_initial_place = int1

                net, counts, int2 = recursively_add_tree(
                    tree,
                    tree_childs[i],
                    net,
                    target_initial_place,
                    int2,
                    counts,
                    rec_depth + 1,
                    global_sink=global_sink,
                    parent_final_place=final_place,
                    global_source=global_source,
                    bypasses=bypasses,
                )

            # NOTE: EnhancedProcessTree(), not ProcessTree() -- the recursion
            # reads .start/.stop/.skip directly, so the silent exit leaf has to
            # carry those attributes too.
            net, counts, int3 = recursively_add_tree(
                tree,
                EnhancedProcessTree(),
                net,
                int1,
                final_place,
                counts,
                rec_depth + 1,
                global_sink=global_sink,
                parent_final_place=final_place,
                global_source=global_source,
                bypasses=bypasses,
            )

            looping_place = int2

            add_arc_from_to(looping_place, loop_trans, net)
            add_arc_from_to(loop_trans, initial_place, net)

    # A stop annotation creates a bypass straight to the global sink
    if tree.stop and global_sink is not None and final_place != global_sink:
        tau_stop = get_new_hidden_trans(counts, type_trans="tau_stop")
        net.transitions.add(tau_stop)

        add_arc_from_to(final_place, tau_stop, net)
        add_arc_from_to(tau_stop, global_sink, net)

        bypasses.append(Bypass(tau_stop, "stop", global_sink))

    # A skip annotation creates a bypass to the final place of the parent
    elif (
        tree.skip
        and parent_final_place is not None
        and final_place != parent_final_place
    ):
        tau_skip = get_new_hidden_trans(counts, type_trans="tau_skip")
        net.transitions.add(tau_skip)

        add_arc_from_to(final_place, tau_skip, net)
        add_arc_from_to(tau_skip, parent_final_place, net)

        bypasses.append(Bypass(tau_skip, "skip", parent_final_place))

    return net, counts, final_place


def apply(tree, parameters=None):
    """
    Applies the conversion from an enhanced process tree to a reset net

    Parameters
    -----------
    tree
        Enhanced process tree
    parameters
        Parameters of the algorithm:
            - Parameters.APPLY_REDUCTION: whether the simple Petri net reduction
              should be applied afterwards (default: True). The reduction is not
              aware of reset arcs, so disable it if the resulting net contains
              reset arcs whose semantics must be preserved exactly.
            - Parameters.REMOVE_DANGLING_PLACES: whether places without incoming
              or outgoing arcs should be removed (default: True)

    Returns
    -----------
    net
        Reset net
    initial_marking
        Initial marking
    final_marking
        Final marking
    """
    if parameters is None:
        parameters = {}

    if not is_enhanced(tree):
        raise TypeError(
            "the conversion to a reset net requires an EnhancedProcessTree "
            "in which every node is an EnhancedProcessTree; got a tree rooted "
            "in %s. Use pm4py.objects.enhanced_process_tree.utils.generic."
            "from_process_tree to convert a plain process tree."
            % type(tree).__name__
        )

    apply_reduction = exec_utils.get_param_value(
        Parameters.APPLY_REDUCTION, parameters, True
    )
    remove_dangling_places = exec_utils.get_param_value(
        Parameters.REMOVE_DANGLING_PLACES, parameters, True
    )

    counts = Counts()
    net = ResetNet("enhanced_pt_net_" + str(time.time()))
    initial_marking = Marking()
    final_marking = Marking()
    source = get_new_place(counts)
    source.name = "source"
    sink = get_new_place(counts)
    sink.name = "sink"
    net.places.add(source)
    net.places.add(sink)
    initial_marking[source] = 1
    final_marking[sink] = 1

    initial_mandatory = check_tau_mandatory_at_initial_marking(tree)
    final_mandatory = check_tau_mandatory_at_final_marking(tree)

    if initial_mandatory:
        initial_place = get_new_place(counts)
        net.places.add(initial_place)
        tau_initial = get_new_hidden_trans(counts, type_trans="tau")
        net.transitions.add(tau_initial)
        add_arc_from_to(source, tau_initial, net)
        add_arc_from_to(tau_initial, initial_place, net)
    else:
        initial_place = source

    if final_mandatory:
        final_place = get_new_place(counts)
        net.places.add(final_place)
        tau_final = get_new_hidden_trans(counts, type_trans="tau")
        net.transitions.add(tau_final)
        add_arc_from_to(final_place, tau_final, net)
        add_arc_from_to(tau_final, sink, net)
    else:
        final_place = sink

    net, counts, last_added_place = recursively_add_tree(
        tree,
        tree,
        net,
        initial_place,
        final_place,
        counts,
        0,
        global_sink=sink,
        parent_final_place=final_place,
        global_source=source,
        bypasses=[],
    )

    if apply_reduction:
        reduction.apply_simple_reduction(net)

    if remove_dangling_places:
        places = list(net.places)
        for place in places:
            if len(place.out_arcs) == 0 and place not in final_marking:
                remove_place(net, place)
            if len(place.in_arcs) == 0 and place not in initial_marking:
                remove_place(net, place)

    return net, initial_marking, final_marking