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
from enum import Enum
from typing import Optional, Dict, Any, Union

import pandas as pd

from pm4py import util as pmutil
from pm4py.algo.discovery.enhanced_process_tree.variants.log_refinement import LogRefinement
from pm4py.algo.discovery.enhanced_process_tree.variants.skip_point_post_processor import SkipPointPostProcessor
from pm4py.algo.discovery.enhanced_process_tree.variants.tree_post_processor import TreePostProcessor
from pm4py.algo.discovery.inductive.algorithm import tree_sort
from pm4py.algo.discovery.inductive.dtypes.im_ds import (
    IMDataStructureUVCL,
)
from pm4py.algo.discovery.inductive.variants.im import IMUVCL
from pm4py.algo.discovery.inductive.variants.imf import IMFUVCL
from pm4py.objects.enhanced_process_tree.obj import EnhancedProcessTree
from pm4py.objects.enhanced_process_tree.utils.generic import convert_to_enhanced_process_tree
from pm4py.objects.log.obj import EventLog
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.objects.process_tree.utils import generic as pt_util
from pm4py.util import constants, exec_utils
from pm4py.util import xes_constants as xes_util
from pm4py.util.compression import util as comut
from pm4py.util.compression.dtypes import UVCL


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY
    LIMIT = "limit"
    NOISE_THRESHOLD = "noise_threshold"
    ALIGNMENT_THRESHOLD = "alignment_threshold"
    TAU_DELETION_THRESHOLD = "tau_deletion_threshold"
    VARIANT_THRESHOLD = "variant_threshold"
    SKIP_COVERAGE_THRESHOLD = "skip_coverage_threshold"
    OPTIMIZE_PARALLEL_SEQUENCES = "optimize_parallel_sequences"


class Variant(Enum):
    """
    Defines the architectural pipeline used to discover the Enhanced Process Tree.
    """
    # Uses Log Refinement to build the tree and inserts skips with the knowledge of the tree.
    REFINEMENT_HYBRID = "refinement_hybrid"

    # Discovers a standard base tree, then runs A* Alignments to find skips.
    ALIGNMENTS = "alignments"

def apply(
        obj: Union[EventLog, pd.DataFrame, UVCL],
        variant: Variant = Variant.REFINEMENT_HYBRID,
        parameters: Optional[Dict[Any, Any]] = None,
) -> EnhancedProcessTree:
    """
    Applies the Enhanced Process Tree discovery algorithm based on the selected variant.
    """
    if parameters is None:
        parameters = {}

    ack = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes_util.DEFAULT_NAME_KEY)
    tk = exec_utils.get_param_value(Parameters.TIMESTAMP_KEY, parameters, xes_util.DEFAULT_TIMESTAMP_KEY)
    cidk = exec_utils.get_param_value(Parameters.CASE_ID_KEY, parameters, pmutil.constants.CASE_CONCEPT_NAME)
    noise_threshold = exec_utils.get_param_value(Parameters.NOISE_THRESHOLD, parameters, 0.0)
    limit = exec_utils.get_param_value(Parameters.LIMIT, parameters, 25)
    variant_threshold = exec_utils.get_param_value(Parameters.VARIANT_THRESHOLD, parameters, 0.0)

    original_log = obj
    refined_event_log = original_log
    refined_dict = {}
    annotations = {}
    skip_records = []

    if variant == Variant.REFINEMENT_HYBRID:
        print("Discovering Enhanced Process Tree trough preprocessing.")
        if isinstance(obj, dict):
            uvcl = obj.copy()
        else:
            uvcl = comut.get_variants(comut.project_univariate(obj, key=ack, df_glue=cidk, df_sorting_criterion_key=tk))

        log_refinement = LogRefinement(uvcl, variant_threshold=variant_threshold)
        refined_dict, annotations, skip_records = log_refinement.run(limit=limit)

        ds_uvcl = IMDataStructureUVCL(refined_dict)
        process_tree = ProcessTree()

        if noise_threshold == 0.0:
            print("Noise threshold is 0.0: Using Standard Inductive Miner (IM).")
            im = IMUVCL(parameters)
            process_tree = im.apply(ds_uvcl, parameters)
        else:
            print(f"Noise threshold is {noise_threshold}: Using Inductive Miner Infrequent (IMf).")
            imf = IMFUVCL(parameters)
            process_tree = imf.apply(ds_uvcl, parameters)

        process_tree = pt_util.fold(process_tree)
        tree_sort(process_tree)

        enhanced_tree = convert_to_enhanced_process_tree(process_tree, annotations)

        print("Adding the Skip annotations.")
        hybrid_processor = SkipPointPostProcessor(parameters=parameters)
        return hybrid_processor.apply(enhanced_tree, refined_dict, skip_records)
    else:
        print("Discovering Enhanced Process Tree trough postprocessing.")
        from pm4py import discover_process_tree_inductive
        process_tree = discover_process_tree_inductive(
            refined_event_log,
            noise_threshold=noise_threshold,
            activity_key=ack,
            timestamp_key=tk,
            case_id_key=cidk
        )
        print("Discovered Process Tree. Convert to Enhanced Process Tree.")
        enhanced_tree = convert_to_enhanced_process_tree(process_tree, annotations)

        alignment_processor = TreePostProcessor(parameters=parameters)
        return alignment_processor.apply(original_log, enhanced_tree)