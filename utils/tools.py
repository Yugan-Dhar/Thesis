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

    if args.baseline_bart_training:
        return "BART_Baseline", 1, previous_results

    version_counter = 1

    model_id = f"{args.extractive_model}_{args.abstractive_model}_{args.mode}"
    if args.mode == "Fixed" or args.mode == "Hybrid":
        model_id += f"_ratio_{args.compression_ratio}"

    model_id += f"_V{version_counter}"

    while any(entry["Model_ID"] == model_id for entry in previous_results):
        version_counter += 1
        model_id = f"{args.extractive_model}_{args.abstractive_model}_{args.mode}"
        if args.mode == "Fixed" or args.mode == "Hybrid":
            model_id += f"_ratio_{args.compression_ratio}"
        model_id += f"_V{version_counter}"

    return model_id, version_counter, previous_results


def calculate_hybrid_final_step_ratio(intermediate_summary, abstractive_model_token_length, extractive_tokenizer):
     
    token_length = token_length = extractive_tokenizer(intermediate_summary, return_tensors='pt')['input_ids'].shape[1]
    final_ratio = (abstractive_model_token_length / token_length)

    #Disabled this because it shouldn't be necessary!! If the ratio is larger than 1, then something is going wrong.
    #If the ratio is larger than 1, it means that the ab. token length is larger than the text token length, which should not be possible as it's the last step of extractive summarization
    # i.e. because the steps are always rouned up it means that the ratio for the final step is always between 0 and 1. 
    """if final_ratio > 1:
        final_ratio = 1"""
    return final_ratio