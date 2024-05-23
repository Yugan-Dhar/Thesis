#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 -e ex_models -a ab_models -o modes"
    exit 1
}

# Parse command line arguments
while getopts "e:a:o:" opt; do
    case ${opt} in
        e)
            IFS=',' read -r -a ex_models <<< "${OPTARG}"
            ;;
        a)
            IFS=',' read -r -a ab_models <<< "${OPTARG}"
            ;;
        o)
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

# Open vastai instance
vastai start instance 10937241

sleep 5m

# ssh into vastai instance and run the commands in the here document sequentially
ssh -p 40033 root@server-ip -L 8080:localhost:8080 << EOF

# Activate the correct environment
cd workspace/Thesis
source venv/bin/activate

# Run the training.py script with different parameters
for i in ${!ex_models[@]}; do
    python3 training.py ${ex_models[$i]} 4 ${ab_models[$i]} -m ${modes[$i]}
    
    # Add and commit results to git repository after each run
    git add .
    git commit -m "Auto commit after running with ${ex_models[$i]}, ${ab_models[$i]}, ${modes[$i]}"
    git push
done

# exit ssh connection
exit
EOF

# close vastai instance
vastai stop instance 10937241