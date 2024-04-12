import os
import json

def get_id_and_version_and_prev_results(evaluation_results_filepath, args):
    """
    Generates a unique model ID and version number based on the existing json file.

    Args:
        previous_results (list): A list of dictionaries containing previous model results.

    Returns:
        tuple: A tuple containing the generated model ID and the version number.

    """
    if os.path.isfile(evaluation_results_filepath):
        with open(evaluation_results_filepath, 'r') as f:
            previous_results = json.load(f)
    else:
            previous_results = []

    version_counter = 1
    #TODO: Change model_id to include different ratio types. If it is a fixed type then it should be fixed_ratio_0{args.compression_ratio}
    # Add more if statemetns to determine the different types of ratios. Because we only want to include the actual ratio number if it is a fixed ratio.
    model_id = f"{args.abstractive_model}_{args.extractive_model}_ratio_0{args.compression_ratio}_V{version_counter}"

    while any(entry["Model_ID"] == model_id for entry in previous_results):
        version_counter += 1
        model_id = f"{args.abstractive_model}_{args.extractive_model}_ratio_0{args.compression_ratio}_V{version_counter}"

    return model_id, version_counter, previous_results


def calculate_hybrid_final_step_ratio(intermediate_summary, abstractive_model_token_length, extractive_tokenizer):
     
    token_length = token_length = extractive_tokenizer(intermediate_summary, return_tensors='pt')['input_ids'].shape[1]
    return (abstractive_model_token_length / token_length)