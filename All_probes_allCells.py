import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

def read_and_plot_etype_data(base_folder, csv_path):
    """
    Process and plot data based on etypes and their corresponding folders and CSVs.
    
    Parameters:
        base_folder (str): The path to the folder containing etype-specific subfolders.
        csv_path (str): The path to the CSV file containing 'etype' information.
    """
    # Read the main CSV to get etypes
    df = pd.read_csv(csv_path)
    etypes = df['etype'].unique()

    # Loop over each etype to process corresponding folders
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for etype in etypes:
        etype_folder_path = os.path.join(base_folder, f'*{etype}*')
        etype_folders = glob.glob(etype_folder_path)  # Match all folders containing the etype

        # Data dictionary to hold data frames for soma, axon, dend, api
        data = {'soma': [], 'axon': [], 'dend': [], 'api': []}

        # Process each folder matched for the current etype
        for folder in etype_folders:
            csv_files = glob.glob(os.path.join(folder, '*.csv'))  # All CSV files in the folder
            
            # Read each CSV and categorize it
            for file_path in csv_files:
                category = None
                if 'soma' in file_path:
                    category = 'soma'
                elif 'axon' in file_path:
                    category = 'axon'
                elif 'dend' in file_path:
                    category = 'dend'
                elif 'api' in file_path:
                    category = 'api'
                
                if category:
                    mean_ecd = pd.read_csv(file_path)['Mean ECD'].values  # Assuming 'Mean ECD' column exists
                    data[category].append(mean_ecd)
        
        # Plot data for this etype
         # 4 subplots for soma, axon, dend, api
        axs = axs.ravel()
        categories = ['soma', 'axon', 'dend', 'api']

        for i, cat in enumerate(categories):
            if data[cat]:  # Check if there is any data collected for the category
                combined_data = pd.DataFrame(data[cat]).T  # Transpose to align the data correctly
                axs[i].boxplot(combined_data)
                axs[i].set_title(f'{etype} - {cat.capitalize()}')
                axs[i].set_xlabel('Parameters')
                axs[i].set_ylabel('Mean ECD')
        
    plt.tight_layout()
    plt.savefig("AllInh.png")

# Usage
base_folder_path = '/global/cfs/projectdirs/m2043/roybens/sens_ana/InhSens_all44'
csv_file_path = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/InhibitoryCell1.csv'
read_and_plot_etype_data(base_folder_path, csv_file_path)
