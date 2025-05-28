import argparse
import os
import re
from typing import Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm
from vllm import LLM, SamplingParams

from utils._path import DATA_SOURCE, DATABASE
from utils.json_tools import save_json


def load_model(model_name: str, device: str = 'cuda') -> LLM:
    """
    Load the specified LLM model.

    Args:
        model_name (str): The name of the model to load.
        device (str): The device to run the model on (e.g., 'cuda', 'cpu').

    Returns:
        LLM: An instance of the loaded model.
    """
    return LLM(
        model=model_name,
        device=device,
        dtype="bfloat16",
        trust_remote_code=True,
        quantization="bitsandbytes",
        max_model_len=8192,
    )


def generate_text(
        model: LLM,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
        max_tokens: int = 512,
    ) -> str:
    """
    Generate triplets using the specified model and prompt.
    Args:
        model (LLM): The LLM model to use for text generation.
        prompt (str): The input prompt for the model.
        temperature (float): Sampling temperature for text generation.
        top_p (float): Top-p sampling parameter for text generation.
        max_tokens (int): Maximum number of tokens to generate.
    Returns:
        str: The generated triplets from the model.
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_tokens=max_tokens
    )
    responses = model.chat(prompt, sampling_params=sampling_params)
    return responses[0].outputs[0].text.strip() if responses else ""


def get_pdf_paths(data_source_dir: str) -> List[str]:
    """
    Load all PDF files from the specified directory.
    Args:
        data_source_dir (str): The directory containing the PDF files.
    Returns:
        List[str]: A list of file paths to the PDF files.
    """
    # List all PDF files in the data source directory
    full_data_source_dir = os.path.join(DATA_SOURCE, data_source_dir)
    print(f"Looking for PDF files in: {full_data_source_dir}")
    if not os.path.exists(full_data_source_dir):
        raise FileNotFoundError(f"The directory {full_data_source_dir} does not exist.")
    pdf_files = [os.path.join(full_data_source_dir, f) for f in os.listdir(full_data_source_dir) if f.endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in the directory {full_data_source_dir}.")
    return pdf_files


def load_pdf_files(file_paths: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Load a PDF file and split it into chunks.

    Args:
        file_paths (List[str]): List of paths to the PDF files.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        List[str]: A list of text chunks from the PDF.
    """
    text_chunks = []
    for path in file_paths:
        print(f"Processing file: {path}")
        loader = PyPDFLoader(path)
        text_chunks.extend(loader.load())

    # Split the loaded documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_pieces = text_splitter.split_documents(text_chunks)
    text_pieces = list(map(lambda x: re.sub('\n', ' ', re.sub('-\n', '', x.page_content)), text_pieces))

    return text_pieces


def prepare_prompts(text_pieces: List[str]) -> List[List[dict]]:
    """
    Prepare prompts for the LLM based on the text pieces.

    Args:
        text_pieces (List[str]): List of text pieces to create prompts from.

    Returns:
        List[List[dict]]: A list of prompts formatted for the LLM.
    """
    prompts = []
    for text in text_pieces:
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides information based on the input provided."
            },
            {
                "role": "user",
                "content": f'Please extract all triplet relations in the format \n(subject, relation, object)\n from the following text. If there are no such relations, respond with only "None". Do not output any explanations or additional text.\n\n text: {text}'
            }
        ]
        prompts.append(prompt)
    return prompts



def clean_triplet(generated_relations: List[str]) -> List[Dict]:
    """
    Clean the generated triplet relations and format them into a structured list.
    Args:
        generated_relations (List[str]): List of generated relations from the LLM.
    Returns:
        List[Dict]: A list of dictionaries containing the text chunk index and the extracted triplets.
    """
    result = []
    for index, relation in enumerate(generated_relations):
        triplets = re.findall(r'\(.*?, .*?, .*?\)', relation)
        if len(triplets) == 0 or (len(triplets) == 1 and triplets[0] == '(None)'):
            continue
        triplets = [triplet.strip('()').split(',') for triplet in triplets]
        triplets = [tuple(map(str.strip, triplet)) for triplet in triplets if len(triplet) == 3]
        result.append({
            "text_chunk_index": index,
            "triplets": triplets
        })

    return result


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="LLM Triplet Relation Extraction")
    parser.add_argument('--data_source_dir', type=str,
                        help='Directory containing the data source PDF files')
    parser.add_argument('--database_dir', type=str,
                        help='Directory to store the database files')
    parser.add_argument('--model_name', type=str, default='microsoft/phi-4',
                        help='Name of the LLM model to use')
    parser.add_argument('--chunk_size', type=int, default=512, help='Size of text chunks to process')
    parser.add_argument('--chunk_overlap', type=int, default=256, help='Overlap size between text chunks')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum tokens to generate in response')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for text generation')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling parameter for text generation')
    parser.add_argument('--top_k', type=int, default=-1, help='Top-k sampling parameter for text generation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (e.g., cuda, cpu)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load the PDF files from the specified directory
    pdf_paths = get_pdf_paths(data_source_dir=args.data_source_dir)
    text_pieces = load_pdf_files(
        file_paths=pdf_paths,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    # Prepare the prompt for the LLM
    prompts = prepare_prompts(text_pieces)

    # Generate triplet relations from the text pieces
    generated_relations = []
    model = load_model(model_name=args.model_name, device=args.device)
    for prompt in tqdm(prompts, desc="Generating text"):
        generated_text = generate_text(
            model=model,
            prompt=prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens
        )
        generated_relations.append(generated_text)

    # Check if the database directory exists, if not create it
    full_database_dir = os.path.join(DATABASE, args.database_dir)
    if not os.path.exists(full_database_dir):
        os.makedirs(full_database_dir)

    # Save original text pieces to a JSON file
    text_pieces_file = os.path.join(full_database_dir, 'text_pieces.json')
    save_json(text_pieces, text_pieces_file)

    # Clean the generated triplet relations
    cleaned_triplets = clean_triplet(generated_relations)
    cleaned_triplets_file = os.path.join(full_database_dir, 'cleaned_triplets.json')
    save_json(cleaned_triplets, cleaned_triplets_file)


if __name__ == "__main__":
    main()