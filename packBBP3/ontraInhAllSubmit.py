import pandas as pd
import os
import subprocess
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=False, help="Path to the CSV file",default="/global/homes/k/ktub1999/mainDL4/DL4neurons2/Grouped_PMJobs.csv")
    parser.add_argument("--path", type=str, required=False, help="Path to the simulation files",default="/pscratch/sd/k/ktub1999/BBP_Inh_Feb5thAll150CellsNoNoise/runs2")
    parser.add_argument("--outPath_base", type=str, required=False, help="Base output path",default="/pscratch/sd/k/ktub1999/BBP_Ontra_Inhibitory_Exclude")
    args = parser.parse_args()
    return args




if __name__ == '__main__':
    # Define paths
    args=get_parser()
    csv_file = args.csv_file  # Replace with the actual path to your CSV file
    Path = args.path  # Replace with the actual path to your simulation files
    outPath_base = args.outPath_base  # Replace with the base output path
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Group by eType
    grouped = df.groupby("Processed_mType")

    # Iterate through unique eTypes
    unique_eTypes = df["Processed_mType"].unique()
    unique_eTypes = sorted(unique_eTypes)
    rank = int(os.environ['SLURM_PROCID'])
    n_tasks = int(os.environ['SLURM_NPROCS'])
    m_types_per_node = len(unique_eTypes)//n_tasks
    if(rank == n_tasks-1):
        m_types_per_node = len(unique_eTypes) - (n_tasks-1)*m_types_per_node
    start_index = rank * m_types_per_node
    end_index = start_index + m_types_per_node
    if end_index > len(unique_eTypes):
        end_index = len(unique_eTypes)
    print(f"Rank {rank} processing eTypes from index {start_index} to {end_index}")
    unique_eTypes = unique_eTypes[start_index:end_index]
    for eType_exclude in unique_eTypes:
        ALL_Cells_Inhibitory_Intrapolated = []
        ALL_CELLS_Inhibitory = []
        ALL_CELLS_Inhibitory_Extrapolation = []

        for eType, group in grouped:
            job_ids = group["JobId"].tolist()

            if eType == eType_exclude:
                # Add all job IDs for the excluded eType
                ALL_CELLS_Inhibitory_Extrapolation.extend(job_ids)
            else:
                # Add job IDs to Intrapolated list until it has size 3
                while len(ALL_Cells_Inhibitory_Intrapolated) < 3 and len(job_ids)>1:
                    ALL_Cells_Inhibitory_Intrapolated.append(job_ids.pop(0))
                # Add remaining job IDs to Inhibitory list
                ALL_CELLS_Inhibitory.extend(job_ids)

        # Define output paths
        outPath = f"{outPath_base}_{eType_exclude}"

        # Ensure output directory exists
        os.makedirs(outPath, exist_ok=True)

        # Run bash commands for each list
        for job_list, file_name in [
            (ALL_CELLS_Inhibitory, "ALL_CELLS_Inhibitory"),
            (ALL_Cells_Inhibitory_Intrapolated, "ALL_CELLS_Inhibitory_Intrapolated"),
            (ALL_CELLS_Inhibitory_Extrapolation, "ALL_CELLS_Inhibitory_Extrapolation"),
        ]:
            job_ids_str = " ".join(map(str, job_list))
            command = (
                f"python3 aggregate_All65.py --simPath {Path} --outPath {outPath} "
                f"--jid {job_ids_str} --fileName '{file_name}' --probes 0 1 2"
            )
            print(f"Running packer for : {command}")
            
            subprocess.run(command, shell=True, check=True)
            print(f"Packer completed for {file_name} with eType {eType_exclude}")
            thread_total = 2
            if file_name == "ALL_CELLS_Inhibitory":
                thread_total = 10
            command = (
                f"python3 format_bbp3_for_ML_paralelly.py --cellName '{file_name}' --dataPath {outPath} --thread_total {thread_total} "
                # f"--jid {job_ids_str} --fileName '{file_name}' --probes 0 1 2"
            )
            print(f"Running format for : {command}")
            subprocess.run(command, shell=True, check=True)
