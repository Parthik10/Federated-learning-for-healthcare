import argparse
import json
import time
import os
from fl_utils import run_simulation

STATE_FILE = "simulation_state.json"

def write_state(round_num, accuracy, logs, status="RUNNING"):
    data = {"round": round_num, "accuracy": accuracy, "logs": logs, "status": status}
    # Atomic write attempt (write temp then rename) often safest, but direct write ok for demo
    with open(STATE_FILE, "w") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="chest_xray", help="Name of the dataset folder")
    args = parser.parse_args()

    # Init state
    write_state(0, 0.0, [], "INITIALIZING")

    def stats_callback(round_num, accuracy, logs):
        write_state(round_num, accuracy, logs, "RUNNING")

    try:
        run_simulation(args.rounds, stats_callback, args.dataset)
        # Read final state to get last logs/acc
        with open(STATE_FILE, "r") as f:
            final_data = json.load(f)
        write_state(final_data["round"], final_data["accuracy"], final_data["logs"], "COMPLETED")
    except Exception as e:
        write_state(0, 0.0, [f"ERROR: {str(e)}"], "FAILED")
        raise e

if __name__ == "__main__":
    main()
