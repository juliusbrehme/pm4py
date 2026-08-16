import csv
import itertools
import logging
import os
import resource
import sys
import time
import traceback

import pm4py
from benchmarks import phase_timing
from benchmarks.process_runner import run_phased
from pm4py import discover_enhanced_process_tree
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algo
from pm4py.algo.discovery.enhanced_process_tree.algorithm import Variant as EnhancedTreeVariant
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.inhibitor_reset.semantics import InhibitorResetSemantics
from pm4py.objects.petri_net.utils import align_utils

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CONFIGURATION ---
if len(sys.argv) > 1:
    TARGET_FOLDER = os.path.join(SCRIPT_DIR, sys.argv[1])
else:
    TARGET_FOLDER = os.path.join(SCRIPT_DIR, "test_dataset")

DATASET_FOLDER = TARGET_FOLDER
RESULTS_FILE = os.path.join(TARGET_FOLDER, "evaluation_results.csv")  # Save in benchmarks/
LOG_FILE = os.path.join(TARGET_FOLDER, "evaluation_pipeline.log")

# Hard deadlines. The run happens in a child process, so on timeout the worker
# (and anything it spawned) is SIGKILLed -- no CPU keeps burning. None = no limit.
DISCOVERY_TIMEOUT_SECONDS = 10800
EVALUATION_TIMEOUT_SECONDS = 10800

# Define parameter grids
NOISE_THRESHOLDS = [0.0, 0.2, 0.4, 0.6]
OPTIMIZE_PARALLEL_SEQUENCES = [False]

# Preprocessing specific parameters
VARIANT_THRESHOLDS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6]
SKIP_COVERAGE_THRESHOLDS = [1.0, 0.75, 0.5]
LIMIT = [25]  # Use default, static safety limit, no need to grid search

# Postprocessing specific parameters
TAU_DELETION_THRESHOLDS = [1.0, 0.9, 0.8, 0.6, 0.4, 0.2]
ALIGNMENT_THRESHOLDS = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6]

# --- CSV SCHEMA ---
# Columns that make a run unique (used for resuming).
RUN_ID_COLUMNS = [
    "Dataset", "Method", "IM_Noise", "Optimize_Parallel_Sequences",
    "Variant_Threshold", "Skip_Coverage_Threshold", "Limit",
    "Tau_Deletion_Threshold", "Alignment_Threshold",
]

# Phases reported by phase_timing from inside the discovery code.
# Key = name passed to phase_timing.phase(...), value = CSV column.
# Stays empty as long as the discovery code isn't instrumented yet.
PHASE_COLUMNS = {
    "preprocessing": "Preprocessing_Time_s",
    "inductive_miner": "IM_Time_s",
    "postprocessing": "Postprocessing_Time_s",
}

TIME_COLUMNS = (
        ["Discovery_Time_s"]
        + list(PHASE_COLUMNS.values())
        + ["Evaluation_Time_s", "Total_Time_s"]
)

# Add "Simplicity" here (and to evaluate_reset_net) once it is implemented.
METRIC_COLUMNS = ["Fitness", "Avg_Trace_Fitness", "Percentage_of_Fitting_Traces", "Precision", "F1_score",
                  "Language_Size",
                  "Total_Nodes", "Start_Nodes", "Stop_Nodes", "Skip_Nodes"
                  ]

HEADERS = RUN_ID_COLUMNS + ["Status"] + TIME_COLUMNS + METRIC_COLUMNS + ["Error_Message"]

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)


def setup_results_csv(file_path):
    """Creates the CSV with dedicated columns for all specific parameters."""
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as f:
            csv.DictWriter(f, fieldnames=HEADERS).writeheader()


def make_run_id(dataset_name, config):
    """Unique signature of one (dataset, configuration) pair."""
    return tuple([dataset_name] + [str(config[col]) for col in RUN_ID_COLUMNS[1:]])


def get_completed_runs(file_path):
    """Reads the CSV to find out which runs are already done based on their unique signature."""
    completed = set()
    if os.path.exists(file_path):
        with open(file_path, mode='r') as f:
            for row in csv.DictReader(f):
                if row.get("Status") == "SUCCESS":
                    completed.add(tuple(row.get(col, "None") for col in RUN_ID_COLUMNS))
    return completed


# ---------------------------------------------------------------------------
# CONFIGURATIONS
# ---------------------------------------------------------------------------

def generate_configurations():
    """Generates the exact required parameter combinations per method."""
    configs = []

    # Helper to standardise the dictionary structure for all runs
    def make_config(method, noise="None", opt="None", var_thresh="None",
                    skip_thresh="None", limit="None", tau_thresh="None",
                    align_thresh="None"):
        return {
            "Method": method, "IM_Noise": str(noise), "Optimize_Parallel_Sequences": str(opt),
            "Variant_Threshold": str(var_thresh), "Skip_Coverage_Threshold": str(skip_thresh),
            "Limit": str(limit),
            "Tau_Deletion_Threshold": str(tau_thresh), "Alignment_Threshold": str(align_thresh)
        }

    # 1. Baseline IM
    configs.append(make_config("baseline_im", noise=0.0))

    # 2. Baseline IMf
    # for noise in NOISE_THRESHOLDS:
    #     if float(noise) == 0.0:
    #         continue
    #     configs.append(make_config("baseline_imf", noise=noise))

    # 3. Preprocessing Grid
    for noise, opt, var_t, skip_t, limit in itertools.product(
            NOISE_THRESHOLDS, OPTIMIZE_PARALLEL_SEQUENCES, VARIANT_THRESHOLDS,
            SKIP_COVERAGE_THRESHOLDS, LIMIT
    ):
        configs.append(make_config("preprocessing", noise=noise, opt=opt,
                                   var_thresh=var_t, skip_thresh=skip_t, limit=limit))

    # 4. Postprocessing Grid
    # for noise, opt, tau_t, align_t in itertools.product(
    #         NOISE_THRESHOLDS, OPTIMIZE_PARALLEL_SEQUENCES, TAU_DELETION_THRESHOLDS, ALIGNMENT_THRESHOLDS
    # ):
    #     configs.append(make_config("postprocessing", noise=noise, opt=opt,
    #                                tau_thresh=tau_t, align_thresh=align_t))

    return configs


def execute_discovery(config, log):
    """
    Executes the discovery algorithm corresponding to the pipeline configuration.
    """
    method = config["Method"]

    # 1. Baseline Inductive Miner (IM)
    if method == "baseline_im":
        with phase_timing.phase("inductive_miner"):
            return pm4py.discover_process_tree_inductive(log)

    # 2. Baseline Inductive Miner with Info/Noise filtering (IMf)
    elif method == "baseline_imf":
        with phase_timing.phase("inductive_miner"):
            return pm4py.discover_process_tree_inductive(
                log,
                noise_threshold=float(config["IM_Noise"])
            )

    # 3. Preprocessing Pipeline (Refinement Hybrid Variant)
    elif method == "preprocessing":
        return discover_enhanced_process_tree(
            log,
            variant=EnhancedTreeVariant.REFINEMENT_HYBRID,
            noise_threshold=float(config["IM_Noise"]),
            optimize_parallel_sequences=(config["Optimize_Parallel_Sequences"] == 'True'),
            variant_threshold=float(config["Variant_Threshold"]),
            skip_coverage_threshold=float(config["Skip_Coverage_Threshold"]),
            limit=int(config["Limit"]),
        )

    # 4. Postprocessing Pipeline (Alignments Variant)
    elif method == "postprocessing":
        return discover_enhanced_process_tree(
            log,
            variant=EnhancedTreeVariant.ALIGNMENTS,
            noise_threshold=float(config["IM_Noise"]),
            optimize_parallel_sequences=(config["Optimize_Parallel_Sequences"] == 'True'),
            tau_deletion_threshold=float(config["Tau_Deletion_Threshold"]),
            alignment_threshold=float(config["Alignment_Threshold"])
        )
    return None


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate_reset_net(log, net, initial_marking, final_marking):
    """
    Evaluates Fitness and Precision using the custom Reset Net alignment algorithm.
    """

    # ---------------------------------------------------------
    # STEP 1: CALCULATE ALIGNMENTS FOR THE ENTIRE LOG
    # ---------------------------------------------------------
    # We must define the model costs for all transitions (skip tau transitions)
    model_costs = {
        t: align_utils.STD_MODEL_LOG_MOVE_COST if t.label is not None else 0
        for t in net.transitions
    }

    sync_costs = {
        t: align_utils.STD_SYNC_COST
        for t in net.transitions if t.label is not None
    }

    alignment_parameters = {
        "petri_semantics": InhibitorResetSemantics(),
        "model_cost_function": model_costs,
        "sync_cost_function": sync_costs,

        # MUST be True to calculate Replay Fitness: it gives us the
        # "worst-case" cost, i.e. the denominator for fitness.
        "enable_best_worst_cost": True
    }

    with phase_timing.phase("alignments"):
        alignments_result = alignments_algo.apply(
            log,
            net,
            initial_marking,
            final_marking,
            variant=alignments_algo.Variants.VERSION_DIJKSTRA_SEMANTICS,
            parameters=alignment_parameters
        )

    # ---------------------------------------------------------
    # STEP 2: CALCULATE FITNESS
    # ---------------------------------------------------------
    fitness = fitness_evaluator.evaluate(
        alignments_result,
        variant=fitness_evaluator.Variants.ALIGNMENT_BASED
    )

    # ---------------------------------------------------------
    # STEP 3: CALCULATE ALIGNMENT-BASED PRECISION
    # ---------------------------------------------------------
    precision_parameters = {
        "model_cost_function": model_costs,
        "sync_cost_function": sync_costs,
        "enable_best_worst_cost": False,
        "alignment_variant": alignments_algo.Variants.VERSION_DIJKSTRA_SEMANTICS,

        # You must pass the reset semantics into the precision module too,
        # otherwise av(s) ("what is possible next") won't trigger resets.
        "petri_semantics": InhibitorResetSemantics(),
        "debug_level": 0
    }

    with phase_timing.phase("precision"):
        precision = precision_evaluator.apply(
            log,
            net,
            initial_marking,
            final_marking,
            variant=precision_evaluator.Variants.AUTOMATON_AFTER_ALIGN,
            parameters=precision_parameters
        )

    fitness_val = fitness["log_fitness"]
    precision_val = precision

    f1_score = 0.0
    if (fitness_val + precision_val) > 0:
        f1_score = (2 * fitness_val * precision_val) / (fitness_val + precision_val)

    return {
        "Fitness": fitness["log_fitness"],
        "Avg_Trace_Fitness": fitness["average_trace_fitness"],
        "Percentage_of_Fitting_Traces": fitness["percentage_of_fitting_traces"],
        "Precision": precision,
        "F1_score": f1_score,
    }


def tree_to_net(tree):
    """
    Converts a (possibly enhanced) process tree into the net used for evaluation.
    Enhanced trees come out as ResetNets via the patched converter; baseline
    trees come out as plain Petri nets, which the reset semantics handles fine
    because they simply carry no reset/inhibitor arcs.
    """
    return pm4py.convert_to_petri_net(tree)


def calculate_language_size(net, initial_marking, final_marking, num_traces=1000):
    """
    Approximates the language size by simulating the net and counting unique variants.
    Because pm4py.play_out natively detects ResetNets, it automatically applies
    the correct InhibitorResetSemantics.
    """
    # We use the string keys that PM4Py expects internally
    parameters = {
        "noTraces": num_traces,
        "maxTraceLength": 1000  # Prevent infinite loops from hanging the process
    }

    with phase_timing.phase("playout"):
        simulated_log = pm4py.play_out(net, initial_marking, final_marking, parameters=parameters)

    variants = pm4py.get_variants(simulated_log)

    return len(variants)


def count_tree_nodes(node):
    """
    Recursively walks the process tree to count the total number of nodes,
    as well as the specific enhanced annotations (start, stop, skip).
    """
    counts = {
        "Total_Nodes": 0,
        "Start_Nodes": 0,
        "Stop_Nodes": 0,
        "Skip_Nodes": 0
    }

    def traverse(n):
        counts["Total_Nodes"] += 1

        if getattr(n, 'start', False):
            counts["Start_Nodes"] += 1
        if getattr(n, 'stop', False):
            counts["Stop_Nodes"] += 1
        if getattr(n, 'skip', False):
            counts["Skip_Nodes"] += 1

        for child in n.children:
            traverse(child)

    traverse(node)
    return counts


def evaluate_tree_generic(tree, log):
    """Evaluate the discovered tree: convert to net, then fitness + precision."""
    node_metrics = count_tree_nodes(tree)
    net, initial_marking, final_marking = tree_to_net(tree)
    language_size = calculate_language_size(net, initial_marking, final_marking)
    metrics = evaluate_reset_net(log, net, initial_marking, final_marking)
    metrics["Language_Size"] = language_size
    metrics.update(node_metrics)

    return metrics


# ---------------------------------------------------------------------------
# WORKER (runs in the killable child process)
# ---------------------------------------------------------------------------

# Loaded once per dataset in the parent; children inherit it for free via fork.
_LOG = None
_LOG_PATH = None


def load_log(dataset_path):
    global _LOG, _LOG_PATH
    if _LOG is None or _LOG_PATH != dataset_path:
        _LOG = xes_importer.apply(dataset_path)
        _LOG_PATH = dataset_path
    return _LOG


def _format_error(exc):
    tb = traceback.format_exc().strip().splitlines()
    where = tb[-3].strip() if len(tb) >= 3 else ""
    return f"{type(exc).__name__}: {exc} | {where}".replace("\n", " ")


def _job(queue, config, dataset_path):
    """
    Runs inside the child process. Reports phase timings and metrics back
    through the queue; the parent kills this process if a phase overruns.
    """
    try:
        log = load_log(dataset_path)
    except Exception as e:
        queue.put(("error", "discovery", f"log loading failed: {e}"))
        return

    # --- SET DISCOVERY CPU LIMIT ---
    if DISCOVERY_TIMEOUT_SECONDS:
        _, hard = resource.getrlimit(resource.RLIMIT_CPU)
        resource.setrlimit(resource.RLIMIT_CPU, (DISCOVERY_TIMEOUT_SECONDS, hard))

    phase_timing.reset()
    start = time.perf_counter()
    try:
        tree = execute_discovery(config, log)
    except Exception as e:
        queue.put(("error", "discovery", _format_error(e)))
        return
    discovery_time = time.perf_counter() - start
    queue.put(("phase", "discovery", discovery_time, phase_timing.snapshot()))

    if tree is None:
        queue.put(("error", "discovery", "Discovery returned None"))
        return

    # --- SET EVALUATION CPU LIMIT ---
    if EVALUATION_TIMEOUT_SECONDS:
        # CPU limits are cumulative for the process lifetime.
        # We find out how much CPU we've used so far, and add the evaluation time to it.
        usage = resource.getrusage(resource.RUSAGE_SELF)
        current_cpu_used = int(usage.ru_utime + usage.ru_stime)
        new_limit = current_cpu_used + EVALUATION_TIMEOUT_SECONDS

        _, hard = resource.getrlimit(resource.RLIMIT_CPU)

        if hard != resource.RLIM_INFINITY:
            new_limit = min(new_limit, hard)

        resource.setrlimit(resource.RLIMIT_CPU, (new_limit, hard))

    phase_timing.reset()
    start = time.perf_counter()
    try:
        metrics = evaluate_tree_generic(tree, log)
    except Exception as e:
        queue.put(("error", "evaluation", _format_error(e)))
        return
    queue.put(("phase", "evaluation", time.perf_counter() - start, phase_timing.snapshot()))
    queue.put(("result", metrics))


# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------

def _fmt_time(seconds):
    return "" if seconds is None else f"{seconds:.4f}"


def write_result_row(run_id, outcome):
    status = outcome["status"]
    if status != "SUCCESS":
        status = f"{status}_{(outcome['failed_phase'] or '').upper()}".rstrip("_")

    phase_times = outcome["phase_times"]
    discovery_time = phase_times.get("discovery")
    evaluation_time = phase_times.get("evaluation")
    inner = outcome["extras"].get("discovery", {})
    metrics = outcome["result"] or {}

    row = dict(zip(RUN_ID_COLUMNS, run_id))
    row["Status"] = status
    row["Discovery_Time_s"] = _fmt_time(discovery_time)
    for phase_name, column in PHASE_COLUMNS.items():
        row[column] = _fmt_time(inner.get(phase_name))
    row["Evaluation_Time_s"] = _fmt_time(evaluation_time)
    if discovery_time is not None or evaluation_time is not None:
        row["Total_Time_s"] = _fmt_time((discovery_time or 0.0) + (evaluation_time or 0.0))
    else:
        row["Total_Time_s"] = ""
    for column in METRIC_COLUMNS:
        row[column] = metrics.get(column, "")
    row["Error_Message"] = outcome["error"]

    with open(RESULTS_FILE, mode='a', newline='') as f:
        csv.DictWriter(f, fieldnames=HEADERS, restval="").writerow(row)

    if status == "SUCCESS":
        im_time = inner.get("inductive_miner")
        im_part = f" (IM {im_time:.2f}s)" if im_time is not None else ""
        logging.info(
            f"  -> fitness={metrics['Fitness']:.4f} precision={metrics['Precision']:.4f} "
            f"| discovery {discovery_time:.2f}s{im_part} | evaluation {evaluation_time:.2f}s"
        )
    else:
        logging.warning(f"  -> {status}: {outcome['error']}")


def run_pipeline():
    setup_results_csv(RESULTS_FILE)
    completed_runs = get_completed_runs(RESULTS_FILE)

    datasets = sorted(f for f in os.listdir(DATASET_FOLDER) if f.endswith('.xes'))
    if not datasets:
        logging.warning(f"No .xes files found in '{DATASET_FOLDER}'.")
        return

    configurations = generate_configurations()
    total_tasks = len(datasets) * len(configurations)

    logging.info(f"Found {len(datasets)} datasets. Generating {len(configurations)} configs per dataset.")
    logging.info(f"Total tasks: {total_tasks}. Completed so far: {len(completed_runs)}.")

    for dataset_name in datasets:
        dataset_path = os.path.join(DATASET_FOLDER, dataset_name)

        logging.info(f"Loading dataset: {dataset_name}...")
        try:
            load_log(dataset_path)  # parent loads once, children inherit it via fork
        except Exception as e:
            logging.error(f"Failed to load dataset {dataset_name}: {e}")
            continue

        for config in configurations:
            run_id = make_run_id(dataset_name, config)
            if run_id in completed_runs:
                continue

            logging.info(f"Running: {dataset_name} | {config['Method']} | Params: {config}")

            outcome = run_phased(
                _job,
                args=(config, dataset_path),
                phases=(("discovery", None),
                        ("evaluation", None)),
            )

            write_result_row(run_id, outcome)
            completed_runs.add(run_id)

    logging.info("Evaluation Pipeline Completed!")


if __name__ == "__main__":
    run_pipeline()
