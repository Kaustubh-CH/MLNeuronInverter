#!/bin/bash
search_dir=$1
python_dir="yamlMaker.py"
outFolder=$2
defDesig=$3
dataFile=$4
process_yaml() {
    # Process the YAML file using the Python script
    python3 "$python_dir" --yaml-file "$1" --out $outFolder --default-file $defDesig
    echo $1
    YAML_FILE_NAME=$(basename "$1" | cut -d '.' -f 1)
    echo YAML
    echo $outFolder/$YAML_FILE_NAME
    sbatch batchShifter.slr $dataFile $outFolder/$YAML_FILE_NAME
    # Delete the processed YAML file
    rm "$1"
}



while true; do
    # Find all YAML files in the directory
    yaml_files=$(find "$search_dir" -maxdepth 1 -type f -name "*.yaml" -print)

    # Process each YAML file if found
    if [ -n "$yaml_files" ]; then
        for file in $yaml_files; do
            process_yaml "$file"
        done
    fi

    # Add a delay (adjust the sleep time as needed)
    sleep 10
done