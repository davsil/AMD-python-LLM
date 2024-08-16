import os
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_models(model1_name, model2_name):
    """
    Initialize and return two language models based on the provided model names.

    Args:
        model1_name (str): The name or path of the first model.
        model2_name (str): The name or path of the second model.

    Returns:
        model1: The first language model.
        model2: The second language model.
    """

    tokenizer = AutoTokenizer.from_pretrained(model1_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = AutoModelForCausalLM.from_pretrained(model1_name, torch_dtype=torch.bfloat16).to(device)
    model2 = AutoModelForCausalLM.from_pretrained(model2_name, torch_dtype=torch.bfloat16).to(device)

    return model1, model2


def ask_llama(model, tokenizer, query):

    context = ""
    max_tokens = 16384 

    messages = [
        {"role": "system", "content": "You are a helpful assistant for AMD technologies and python."},
        {"role": "user", "content": query}
    ]

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    getout = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(getout, skip_special_tokens=True)
    return response


def run_consecutively(df, model1, model2, model1_name, model2_name, tokenizer1, tokenizer2):
    """
    For each question, submit to the first model and then the second model.

    Args:
        df (pd.DataFrame): DataFrame containing questions.
        model1: The first language model.
        model2: The second language model.
        model1_name (str): The name of the first model.
        model2_name (str): The name of the second model.
        tokenizer1: tokenizer to use for model1 
        tokenizer2: tokenizer to use for model2 

    Returns:
        pd.DataFrame: DataFrame with responses from both models.
    """

    # Loop through DataFrame and get responses from both models 
    qnum = 0
    subfile = args.output_file.replace(".csv", "_sub.csv")
    field1 = f"Response_{os.path.basename(model1_name)}"
    field2 = f"Response_{os.path.basename(model2_name)}"

    for index, row in df.iterrows():
        qnum = qnum + 1
        question = row['Question']
        print(f"Getting answer for question {index + 1}/{len(df)}: {question}")
        response1 = ask_llama(model1, tokenizer1, question)
        df.at[index, field1] = response1 
        response2 = ask_llama(model2, tokenizer2, question)
        df.at[index, field2] = response2 

        # save intermittent file every 25 questions in case of failure
        if (qnum % 25) == 0:
            df.to_csv(subfile, index=False, sep=';')

    return df


def process_questions(input_file, output_file, model1_name, model2_name):
    """
    Load questions from a CSV file, process them using two language models with formatted prompts,
    and save the results.

    Args:
        input_file (str): Path to the input CSV file containing questions.
        output_file (str): Path to the output CSV file where responses will be saved.
        model1_name (str): The name or path of the first model.
        model2_name (str): The name or path of the second model.

    Returns:
        pd.DataFrame: DataFrame with original questions and model responses.
    """
    df = pd.read_csv(input_file, delimiter=';')
    
    model1, model2 = load_models(model1_name, model2_name)
    tokenizer1 = AutoTokenizer.from_pretrained(model1_name)
    tokenizer2 = AutoTokenizer.from_pretrained(model2_name)

    df = run_consecutively(df, model1, model2, model1_name, model2_name, tokenizer1, tokenizer2)
    df.to_csv(output_file, index=False, sep=';')

    return df


if __name__ == "__main__":
    """
    Main entry point for the script. Handles command-line arguments and runs the process_questions function.
    """
    parser = argparse.ArgumentParser(description="Process questions using two llama3 language models.")
    parser.add_argument("input_file", help="Path to the input CSV file containing questions.")
    parser.add_argument("output_file", help="Path to the output CSV file to save responses.")
    parser.add_argument("model1_name", help="The name or path of the first language model (e.g., LLaMA 3 model).")
    parser.add_argument("model2_name", help="The name or path of the second language model (e.g., LLaMA 3 model).")

    args = parser.parse_args()

    result_df = process_questions(args.input_file, args.output_file, args.model1_name, args.model2_name)
    print(f"Results saved to {args.output_file}")
    print(result_df.head())

