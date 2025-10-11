#!/usr/bin/env python3

import datetime
import os
import sys
import subprocess
from tqdm import tqdm
from enum import Enum, auto
import time
import concurrent.futures

NUM_NPUS = 16

class ParallelismType(Enum):
    RC = auto()
    CR = auto()

class ParallelType(Enum):
    MODEL = auto()
    DATA = auto()
    HYBRID_DATA_MODEL = auto()

class CommunicationType(Enum):
    ALLREDUCE = auto()
    ALLGATHER = auto()
    ALLTOALL = auto()
    NONE = auto()

class CommunicationPhase(Enum):
    FORWARD = auto()
    INPUT = auto()
    WEIGHT = auto()

class NetworkBackendType(Enum):
    ANALYTICAL = auto()
    NS3 = auto()
    GARNET = auto()

PARALLELISM_TYPE: ParallelismType = ParallelismType.RC
COMMUNICATION_TYPE: ParallelType = ParallelType.MODEL
NETWORK_BACKEND_TYPE: NetworkBackendType = NetworkBackendType.ANALYTICAL

class DeepFlowRunner:
    def __init__(self, rundir="DeepFlow"):
        self.rundir = rundir
        self.config_dir = os.path.join(rundir, "configs", "new-configs")
        self.output_dir = os.path.join(rundir, "output")
        self.llm_dir = os.path.join(self.output_dir, "LLM")
        self.input_file = os.path.join(rundir, "scripts", "mat_dims.txt")
        self.num_npus = NUM_NPUS
        self.parallelism_type = PARALLELISM_TYPE
        self.kp1, self.kp2 = self._get_kp1_kp2()

    def prepare_output_directory(self):
        os.makedirs(self.llm_dir, exist_ok=True)
        for root, _, files in os.walk(self.llm_dir):
            for f in files:
                os.remove(os.path.join(root, f))

    def run_all(self):
        self.prepare_output_directory()
        with open(self.input_file, 'r') as f:
            lines = [line.strip().split() for line in f if len(line.strip().split()) >= 3]

        def run_command(idx, parts):
            value1, value2, value3 = parts
            command = [
                "python",
                os.path.join(self.rundir, "perf.py"),
                "--exp_config", os.path.join(self.config_dir, "A100.yaml"),
                "--exp_dir", self.llm_dir,
                "--debug", "False",
                "--id", str(idx),
                "--gemm", "True",
                "--t", self.parallelism_type.name,
                "--dp", "1",
                "--lp", "1",
                "--kp1", f"{self.kp1}",
                "--kp2", f"{self.kp2}",
                "--m", value1,
                "--n", value2,
                "--k", value3
            ]
            subprocess.run(command, stdout=subprocess.DEVNULL)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(tqdm(
                executor.map(lambda args: run_command(*args), enumerate(lines)),
                total=len(lines),
                desc="Running DeepFlow (multithreaded)",
                unit="run"
            ))

    def _get_kp1_kp2(self) -> tuple[int, int]:
        if PARALLELISM_TYPE == ParallelismType.RC:
            if COMMUNICATION_TYPE == ParallelType.MODEL:
                return 1, self.num_npus
            elif COMMUNICATION_TYPE == ParallelType.DATA:
                return self.num_npus, 1
            elif COMMUNICATION_TYPE == ParallelType.HYBRID_DATA_MODEL:
                assert self.num_npus % 2 == 0, "Number of NPUs must be even for hybrid model"
                return 2, int(self.num_npus / 2)
            else:
                sys.exit(1)
        else:
            sys.exit(1)

class LayerMetrics:
    def __init__(self, layername, reserved, fwd_comp_time, fwd_comm_type: CommunicationType, fwd_comm_size,
                 grad_comp_time, grad_comm_type: CommunicationType, grad_comm_size,
                 weight_comp_time, weight_comm_type: CommunicationType, weight_comm_size,
                 final_delay):
        self.layername = layername
        self.reserved = reserved
        self.fwd_comp_time = fwd_comp_time
        self.fwd_comm_type = fwd_comm_type
        self.fwd_comm_size = fwd_comm_size
        self.grad_comp_time = grad_comp_time
        self.grad_comm_type = grad_comm_type
        self.grad_comm_size = grad_comm_size
        self.weight_comp_time = weight_comp_time
        self.weight_comm_type = weight_comm_type
        self.weight_comm_size = weight_comm_size
        self.final_delay = final_delay

    def to_line(self):
        col_widths = [10, 3, 10, 10, 10, 10, 10, 10, 10, 10, 10, 6]
        fields = [
            self.layername.ljust(col_widths[0]),
            str(self.reserved).ljust(col_widths[1]),
            str(self.fwd_comp_time).ljust(col_widths[2]),
            self.fwd_comm_type.name.ljust(col_widths[3]),
            str(self.fwd_comm_size).ljust(col_widths[4]),
            str(self.grad_comp_time).ljust(col_widths[5]),
            self.grad_comm_type.name.ljust(col_widths[6]),
            str(self.grad_comm_size).ljust(col_widths[7]),
            str(self.weight_comp_time).ljust(col_widths[8]),
            self.weight_comm_type.name.ljust(col_widths[9]),
            str(self.weight_comm_size).ljust(col_widths[10]),
            str(self.final_delay).ljust(col_widths[11])
        ]
        return "\t".join(fields)

class DeepFlowParser:
    def __init__(self, rundir="DeepFlow", outdir='astra-sim/'):
        self.rundir = rundir
        self.input_file = os.path.join(rundir, "scripts", "mat_dims.txt")
        self.llm_dir = os.path.join(rundir, "output", "LLM")
        self.out_dir = outdir
        self.output_file = os.path.join(self.out_dir, "examples", "text_converter", "text_workloads", "Df_Model.txt")

    def _get_comm_type(self, comm_type):
        if PARALLELISM_TYPE == ParallelismType.RC:
            if COMMUNICATION_TYPE == ParallelType.MODEL:
                if comm_type == CommunicationPhase.FORWARD:
                    return CommunicationType.ALLGATHER
                elif comm_type == CommunicationPhase.INPUT:
                    return CommunicationType.ALLREDUCE
                elif comm_type == CommunicationPhase.WEIGHT:
                    return CommunicationType.NONE
                else:
                    sys.exit(1)
            elif COMMUNICATION_TYPE == ParallelType.DATA:
                if comm_type == CommunicationPhase.FORWARD:
                    return CommunicationType.NONE
                elif comm_type == CommunicationPhase.INPUT:
                    return CommunicationType.NONE
                elif comm_type == CommunicationPhase.WEIGHT:
                    return CommunicationType.ALLREDUCE
                else:
                    sys.exit(1)
            elif COMMUNICATION_TYPE == ParallelType.HYBRID_DATA_MODEL:
                if comm_type == CommunicationPhase.FORWARD:
                    return CommunicationType.ALLGATHER
                elif comm_type == CommunicationPhase.INPUT:
                    return CommunicationType.ALLREDUCE
                elif comm_type == CommunicationPhase.WEIGHT:
                    return CommunicationType.ALLREDUCE
                else:
                    sys.exit(1)
            else:
                sys.exit(1)
        else:
            sys.exit(1)

    def _extract_float(self, line):
        try:
            return float(line.strip().split(":")[-1].split()[0])
        except ValueError:
            print(f"Error parsing line: {line}")
            return 0.0

    def parse_output_file(self, filepath):
        GB_TO_B = 1024 ** 3

        with open(filepath, 'r') as f:
            lines = f.readlines()

        freq = self._extract_float(lines[5])
        nsec_to_usec = freq / 1000

        fwd_comp_time = int(self._extract_float(lines[9]) * nsec_to_usec)
        fwd_comm_size = self._extract_float(lines[10]) * GB_TO_B

        grad_comp_time = int(self._extract_float(lines[14]) * nsec_to_usec)
        grad_comm_size = self._extract_float(lines[15]) * GB_TO_B

        weight_comp_time = int(self._extract_float(lines[17]) * nsec_to_usec)
        weight_comm_size = self._extract_float(lines[18]) * GB_TO_B

        return LayerMetrics(
            layername="layer1",
            reserved=-1,
            fwd_comp_time=round(fwd_comp_time / 10),
            fwd_comm_type=self._get_comm_type(CommunicationPhase.FORWARD),
            fwd_comm_size=round(fwd_comm_size),
            grad_comp_time=round(grad_comp_time / 10),
            grad_comm_type=self._get_comm_type(CommunicationPhase.INPUT),
            grad_comm_size=round(grad_comm_size),
            weight_comp_time=round(weight_comp_time / 10),
            weight_comm_type=self._get_comm_type(CommunicationPhase.WEIGHT),
            weight_comm_size=round(weight_comm_size),
            final_delay=10
        )

    def parse_all(self):
        with open(self.input_file, 'r') as f:
            valid_lines = [line.strip().split() for line in f if len(line.strip().split()) >= 3]
            # print(valid_lines)
            # sys.exit(1)

        with open(self.output_file, 'w') as out_f:
            out_f.write(f"{COMMUNICATION_TYPE.name}\n")
            out_f.write(f"{len(valid_lines)}\n")

            for idx, (value1, value2, value3) in enumerate(valid_lines):
                expected_file = os.path.join(
                    self.llm_dir, f"detailed_metrics{idx}_m{value1}_n{value2}_k{value3}.txt"
                )
                if os.path.exists(expected_file):
                    metrics = self.parse_output_file(expected_file)
                    # print(metrics.to_line())
                    out_f.write(metrics.to_line() + "\n")
                else:
                    print(f"[\u2717] File doesn't exist: {expected_file}")
        
        print(f"[\u2713] Parsed metrics written to: {self.output_file}")

class AstraSimRunner:
    def __init__(self, project_dir):
        self.project_dir = os.path.abspath(project_dir) # os.path.join(".", project_dir) # os.path.abspath(project_dir)
        self.example_dir = os.path.join(self.project_dir, "examples", "text_converter")
        self.astra_sim_bin = self._get_astra_sim_bin()
        self.system_config = os.path.join(self.example_dir, "system.json")
        self.network_config = self._get_network_config()
        self.logging_config = os.path.join(self.example_dir, "logger.toml")
        self.remote_memory_config = os.path.join(self.example_dir, "remote_memory.json")
        self.logical_topology_config = self._get_logical_topology_config()
        self.target_workload = "Df_Model"  # should match DeepFlowParser.output_file
        self.chakra_converter = "chakra_converter"
        self.num_npus = NUM_NPUS
    
    def _get_astra_sim_bin(self):
        if NETWORK_BACKEND_TYPE == NetworkBackendType.ANALYTICAL:
            return os.path.join(self.project_dir, "build", "astra_analytical", "build", "bin", "AstraSim_Analytical_Congestion_Aware")
        elif NETWORK_BACKEND_TYPE == NetworkBackendType.NS3:
            return os.path.join(self.project_dir, "extern", "network_backend", "ns-3", "build", "scratch", "ns3.42-AstraSimNetwork-default")
        elif NETWORK_BACKEND_TYPE == NetworkBackendType.GARNET:
            sys.exit("[Error] Garnet backend is not supported yet.")
        else:
            sys.exit(f"[Error] Unsupported network backend type: {NETWORK_BACKEND_TYPE.name}")

    def _get_network_config(self):
        if NETWORK_BACKEND_TYPE == NetworkBackendType.ANALYTICAL:
            return os.path.join(self.example_dir, "network.yml")
        elif NETWORK_BACKEND_TYPE == NetworkBackendType.NS3:
            return os.path.join(self.project_dir, "extern", "network_backend", "ns-3", "scratch", "config", "config.txt")
        elif NETWORK_BACKEND_TYPE == NetworkBackendType.GARNET:
            sys.exit("[Error] Garnet backend is not supported yet.")
        else:
            sys.exit(f"[Error] Unsupported network backend type: {NETWORK_BACKEND_TYPE.name}")
    
    def _get_logical_topology_config(self):
        if NETWORK_BACKEND_TYPE == NetworkBackendType.ANALYTICAL:
            return ""
        elif NETWORK_BACKEND_TYPE == NetworkBackendType.NS3:
            return os.path.join(self.project_dir, "examples", "ns3", "sample_8nodes_1D.json")
        elif NETWORK_BACKEND_TYPE == NetworkBackendType.GARNET:
            sys.exit("[Error] Garnet backend is not supported yet.")
        else:
            sys.exit(f"[Error] Unsupported network backend type: {NETWORK_BACKEND_TYPE.name}")
            
    def compile_astrasim(self):
        print(f"[ASTRA-sim] Compiling ASTRA-sim with the {NETWORK_BACKEND_TYPE.name.capitalize()} Network Backend...\n")
        build_script = os.path.join(self.project_dir, "build", f"astra_{NETWORK_BACKEND_TYPE.name.lower()}", "build.sh")

        if NETWORK_BACKEND_TYPE == NetworkBackendType.ANALYTICAL:
            subprocess.run(["bash", build_script], check=True)
        elif NETWORK_BACKEND_TYPE == NetworkBackendType.NS3 or NETWORK_BACKEND_TYPE == NetworkBackendType.GARNET:
            subprocess.run(["bash", build_script, "-c"], check=True)
        else:
            sys.exit(f"[Error] Unsupported network backend type: {NETWORK_BACKEND_TYPE.name}")

        print("[ASTRA-sim] Compilation finished.\n")

    def install_chakra(self):
        print("[ASTRA-sim] Installing Chakra...\n")
        chakra_script = os.path.join(self.project_dir, "utils", "install_chakra.sh")
        subprocess.run(["bash", chakra_script], check=True)
        print("[ASTRA-sim] Chakra installation done.\n")

    def convert_text_to_chakra(self):
        print("[ASTRA-sim] Running Text-to-Chakra converter...\n")
        workload_input = os.path.join(self.example_dir, "text_workloads", f"{self.target_workload}.txt")
        workload_output = os.path.join(self.example_dir, "workload", self.target_workload)
        os.makedirs(os.path.dirname(workload_output), exist_ok=True)

        subprocess.run([
            self.chakra_converter, "Text",
            f"--input={workload_input}",
            f"--output={workload_output}",
            f"--num-npus={self.num_npus}",
            "--num-passes=1"
        ], check=True)

        print("[ASTRA-sim] Text-to-Chakra conversion done.\n")

    def run_astrasim(self):
        log_to_file = False  # Set to False if you don't want to log to a file
        print(f"[ASTRA-sim] Running ASTRA-sim Example with {NETWORK_BACKEND_TYPE.name.capitalize()} Network Backend...\n")
        workload_path = os.path.join(self.example_dir, "workload", self.target_workload)
        # workload_path = "/home/sampan/workflow/pytorch/Torch_model"

        # Timestamped filename
        timestamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"astra_sim_output_{timestamp}.txt"

        # Launch
        bin_dir = os.path.dirname(self.astra_sim_bin)
        backend_args = {
            NetworkBackendType.ANALYTICAL: [
                f"--workload-configuration={workload_path}",
                f"--system-configuration={self.system_config}",
                f"--remote-memory-configuration={self.remote_memory_config}",
                f"--network-configuration={self.network_config}",
            ],
            NetworkBackendType.NS3: [
                f"--workload-configuration={workload_path}",
                f"--system-configuration={self.system_config}",
                f"--remote-memory-configuration={self.remote_memory_config}",
                f"--network-configuration={self.network_config}",
                f"--logical-topology-configuration={self.logical_topology_config}",
                f"--comm-group-configuration=\"empty\""
            ],
            NetworkBackendType.GARNET: None,  # Not supported yet
            # Add more backends here as needed
        }

        cmd_args = [self.astra_sim_bin] + backend_args.get(NETWORK_BACKEND_TYPE, [])

        print(f"[ASTRA-sim] Command: {' '.join(cmd_args)}\n")

        start_time = time.time()  # Start timing

        process = subprocess.Popen(
            cmd_args,
            cwd=bin_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Collect everything
        stdout_data, _ = process.communicate()

        # Always print to terminal
        print(stdout_data, end="", flush=True)

        # Print runtime
        end_time = time.time()  # End timing
        runtime = end_time - start_time
        print(f"\n[ASTRA-sim] Runtime: {runtime:.2f} seconds\n")

        # Optionally write to file
        if log_to_file:
            with open(log_filename, "w") as f:
                f.write(stdout_data)
            print(f"\n[ASTRA-sim] Finished. Log written to: {log_filename}")
        else:
            print(f"\n[ASTRA-sim] Finished. No log file written.")

        # Check exit status
        if process.returncode != 0:
            if log_to_file:
                print(f"[ASTRA-sim] Exited with code {process.returncode} — check {log_filename} for errors.")
            else:
                print(f"[ASTRA-sim] Exited with code {process.returncode}")

    def run_all(self):
        self.compile_astrasim()
        # self.install_chakra()                  #uncomment if you want to install
        # self.convert_text_to_chakra()
        self.run_astrasim()


if __name__ == "__main__":
    # runner = DeepFlowRunner()
    # runner.run_all()

    # parser = DeepFlowParser()
    # parser.parse_all()

    astra_runner = AstraSimRunner(project_dir="astra-sim")  # Adjust if your working directory differs
    astra_runner.run_all()
