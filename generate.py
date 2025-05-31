import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from retrievers.n_hop_retriever import NHopRetriever


def load_model(model_name: str, device: str = 'cpu') -> AutoModelForCausalLM:
    """
    Load a pre-trained model from Hugging Face.

    Args:
        model_name (str): The name of the model to load.
        device (str): The device to load the model on ('cpu' or 'cuda').

    Returns:
        AutoModelForCausalLM: The loaded model.
    """
    bnb_config = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_mode": "symmetric",
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        quantization_config=bnb_config
    )
    return tokenizer, model


def generate_text(
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
        max_new_tokens: int = 512
    ) -> str:
    """
    Generate text using the specified model and prompt.

    Args:
        tokenizer (AutoTokenizer): The tokenizer for the model.
        model (AutoModelForCausalLM): The pre-trained model to use for text generation.
        prompt (str): The input prompt for the model.
        temperature (float): Sampling temperature for text generation.
        top_p (float): Top-p sampling parameter for text generation.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: The generated text from the model.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    model.to('cuda')
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens
    )

    # My RAM is limited to only 24 GB, so I can't move to cpu or my process will crash.
    # If your RAM is large enough, you can uncomment the following lines to free up GPU memory.
    # model.to('cpu')
    # torch.cuda.empty_cache()
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def arg_parser():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model to load.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for text generation.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for text generation.")
    parser.add_argument("--relevant_top_k", type=int, default=5, help="Number of top relevant chunks to retrieve.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter for text generation.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling parameter for text generation (default: `None` for no top-k).")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument("--retriever_ner_model", type=str, default='microsoft/phi-4', help="Path to the NER model for the retriever.")
    parser.add_argument("--retriever_embedding_model", type=str, default='intfloat/multilingual-e5-small', help="Number of hops for the N-hop retriever.")
    parser.add_argument("--retriever_n_hop", type=int, default=1, help="Number of hops for the N-hop retriever.")
    parser.add_argument("--retriever_used_text_chunk", type=bool, default=True, help="Whether to use text chunks in the retriever.")
    return parser.parse_args()


def main():
    args = arg_parser()
    # Initialize the N-hop retriever
    retriever = NHopRetriever(
        NER_model=args.retriever_ner_model,
        embedding_model=args.retriever_embedding_model,
    )

    # Retrieve relevant text chunks based on the prompt
    relevant_information = retriever.retrieve(
        query=args.prompt,
        n_hop=args.retriever_n_hop,
        top_k=args.relevant_top_k,
        use_text_chunk=args.retriever_used_text_chunk,
    )

    # Prepare the prompt with retrieved information
    info = "Relation:\n"
    info += "\n".join([f"{rel_info['relation_text']}" for rel_info in relevant_information])

    if 'text_chunk' in relevant_information[0].keys():
        info += "\n\nText Chunks:\n"
        info += "\n".join([rel_info['text_chunk'] for rel_info in relevant_information])

    # Prepare the template for the model
    template = [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates text based on the provided prompt and relevant information."
        },
        {
            "role": "user",
            "content": f"Below is the relation information:\n{info}\n\n Please answer the following question:\n{args.prompt}"
        }
    ]
    # Load the model and tokenizer
    tokenizer, model = load_model(args.model_name)

    # Generate text based on the provided prompt
    generated_text = generate_text(
        tokenizer=tokenizer,
        model=model,
        prompt=args.prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens
    )

    print("Generated Text:")
    print(generated_text)


    # Example usage of the retriever (assuming you have a corpus to retrieve from)
    # results = retriever.retrieve(generated_text)
    # print("Retrieved Results:")
    # print(results)
    return generated_text

if __name__ == "__main__":
    main()