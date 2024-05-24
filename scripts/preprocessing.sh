#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 -e ex_models -a ab_models -m modes"
    exit 1
}

source venv/bin/activate

# Parse command line arguments
while getopts "e:a:m:" opt; do
    case ${opt} in
        e)
            IFS=',' read -r -a ex_models <<< "${OPTARG}"
            ;;
        a)
            IFS=',' read -r -a ab_models <<< "${OPTARG}"
            ;;
        m)
            IFS=',' read -r -a modes <<< "${OPTARG}"
            ;;
        *)
            usage
            ;;
    esac
done

# Check if all arguments were provided
if [ -z "${ex_models}" ] || [ -z "${ab_models}" ] || [ -z "${modes}" ]; then
    usage
fi

# Ensure that all arrays have the same length
if [ ${#ex_models[@]} -ne ${#ab_models[@]} ] || [ ${#ex_models[@]} -ne ${#modes[@]} ]; then
    echo "Error: All parameter arrays must have the same length"
    exit 1
fi

# ssh into vastai instance and run the commands in the here document sequentially
# Capture the arrays in the remote session
ex_models=(${ex_models[@]})
ab_models=(${ab_models[@]})
modes=(${modes[@]})

# Activate the correct environment

# Run the testing.py script with different parameters
for i in ${!ex_models[@]}; do
    python3 training.py ${ex_models[$i]} 4 ${ab_models[$i]} -m ${modes[$i]} -po -g 1
    
    # Add and commit results to git repository after each run
    echo "Done with ${ex_models[$i]}, ${ab_models[$i]}, ${modes[$i]}"
done


echo "All done for ex_models: ${ex_models[@]}, ab_models: ${ab_models[@]}, modes: ${modes[@]}!"