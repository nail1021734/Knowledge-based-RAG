import torch
from typing import List, Dict
from KG_builder.neo4j import connect_to_neo4j
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb


class NHopRetriever:
    """
    A class to retrieve n-hop relations from a knowledge graph using named entity recognition and semantic embeddings.

    Attributes:
        embedding_model (str): The name of the embedding model to use for semantic embeddings.
        NER_model (str): The name of the LLM model to use for named entity recognition.
        kg_database_driver (GraphDatabase.driver): The driver for connecting to the Neo4j database.
        embedding_model (SentenceTransformer): The loaded embedding model.
        llm_tokenizer (AutoTokenizer): The tokenizer for the LLM model.
        llm_model (AutoModelForCausalLM): The loaded LLM model for named entity recognition.

    Methods:
        get_n_hop_relations(entities: List[str], n_hop: int = 1) -> List[Dict]:
            Retrieve n-hop relations for the given entities from the knowledge graph.
        load_llm_model(model_name: str, device: str = 'cpu') -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
            Load the LLM model for named entity recognition.
        get_named_entities(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text: str) -> List[str]:
            Extract named entities from the text using the specified model.
        load_embedding_model(model_name: str, device: str = 'cpu') -> SentenceTransformer:
            Load the embedding model for semantic embeddings.
        convert_text_to_embedding(model: SentenceTransformer, text: str) -> torch.Tensor:
            Convert text to embeddings using the specified model.
        retrieve(query: str, n_hop: int = 1, top_k: int = 10) -> List[Dict]:
            Retrieve top-k n-hop relations based on the query and named entities.

    Args:
        embedding_model (str): The name of the embedding model to use for semantic embeddings.
        NER_model (str): The name of the LLM model to use for named entity recognition.

    Example:
        retriever = NHopRetriever(
            embedding_model="intfloat/multilingual-e5-small",
            NER_model="microsoft/phi-4"
        )
        results = retriever.retrieve(
            query="What is the LoRA?",
            n_hop=2,
            top_k=10
        )
        print(results)
    """
    def __init__(
        self,
        embedding_model: str,
        NER_model: str,
    ):
        """
        Initialize the NHopRetriever with the specified embedding and NER models.
        Args:
            embedding_model (str): The name of the embedding model to use for semantic embeddings.
            NER_model (str): The name of the LLM model to use for named entity recognition.
        """
        self.embedding_model = embedding_model
        self.llm = NER_model
        self.kg_database_driver = connect_to_neo4j()
        self.embedding_model = self.load_embedding_model(embedding_model)
        self.llm_tokenizer, self.llm_model = self.load_llm_model(NER_model)

    def get_n_hop_relations(
            self,
            entities: List[str],
            n_hop: int = 1,
        ):
        """
        Retrieve n-hop relations for the given entities from the knowledge graph.

        Args:
            entities (List[str]): A list of entities to retrieve relations for.
            n_hop (int): The number of hops to retrieve relations for.
        Returns:
            List[Dict]: A list of dictionaries containing the relations and their properties.
        """
        with self.kg_database_driver.session() as session:
            query = f"""
            MATCH (e:ENTITY)-[r:RELATION*1..{n_hop}]->(o:ENTITY)
            WHERE e.name IN $entities
            RETURN e.name AS subject, r, o.name AS object
            """
            result = session.run(query, entities=entities)
            relations = []
            for record in result:
                relations.append({
                    'subject': record['subject'],
                    'relation': record['r'],
                    'object': record['object']
                })

        return relations


    def load_llm_model(self, model_name: str, device: str = 'cpu'):
        """
        Load the LLM model for named entity recognition.
        Args:
            model_name (str): The name of the LLM model to load.
            device (str): The device to run the model on (e.g., 'cuda', 'cpu').
        Returns:
            Tuple[AutoTokenizer, AutoModelForCausalLM]: A tuple containing the tokenizer and model.
        """
        bnb_config = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_mode": "symmetric",
        }
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization_config=bnb_config,
        )

        return tokenizer, model

    def get_named_entities(
            self,
            model: AutoModelForCausalLM,
            tokenizer: AutoTokenizer,
            text: str
        ) -> List[str]:
        """
        Extract named entities from the text using the specified model.

        Args:
            model (AutoModelForCausalLM): The model to use for named entity recognition.
            tokenizer (AutoTokenizer): The tokenizer for the model.
            text (str): The input text from which to extract named entities.
        Returns:
            List[str]: A list of named entities extracted from the text.
        """
        template = [
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts named entities from text."
            },
            {
                "role": "user",
                "content": f"Extract named entities from the given text and split each entities by commas. Do not include any other text.\n\nText: {text}"
            }
        ]

        prompt = tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=2048)
        inputs = inputs.to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50
        )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        entities = [entity.strip() for entity in response.split(',')]
        return entities if entities else None

    def load_embedding_model(self, model_name: str, device: str = 'cpu') -> SentenceTransformer:
        """
        Load the embedding model for semantic embeddings.

        Args:
            model_name (str): The name of the embedding model to load.
            device (str): The device to run the model on (e.g., 'cuda', 'cpu').
        Returns:
            SentenceTransformer: An instance of the loaded embedding model.
        """
        bnb_config = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_mode": "symmetric",
        }
        return SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={
                "quantization_config": bnb_config,
            },
        )

    def convert_text_to_embedding(
            self,
            model: SentenceTransformer,
            text: str
        ) -> torch.Tensor:
        """
        Convert text to embeddings using the specified model.

        Args:
            model (SentenceTransformer): The embedding model to use.
            text (str): The input text to convert.
        Returns:
            torch.Tensor: The tensor representation of the text embeddings.
        """
        return model.encode(text, convert_to_tensor=True)


    def retrieve(self, query: str, n_hop: int = 1, top_k: int = 10) -> List[Dict]:
        """
        Retrieve top-k n-hop relations based on the query and named entities.
        Args:
            query (str): The input query to retrieve relations for.
            n_hop (int): The number of hops to retrieve relations for.
            top_k (int): The number of top relations to return.
        Returns:
            List[Dict]: A list of dictionaries containing the top-k relations and their scores.
        """
        # Get named entities from the query using the LLM model
        self.llm_model.to('cuda')
        entities = self.get_named_entities(
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            text=query
        )
        self.llm_model.to('cpu')
        torch.cuda.empty_cache()
        if not entities:
            return None
        # Get n-hop relations for the entities
        relations = self.get_n_hop_relations(entities=entities, n_hop=n_hop)

        self.embedding_model.to('cuda')
        # Convert query to embedding
        if self.embedding_model == "intfloat/multilingual-e5-small":
            query = "passage: " + query
        query_embedding = self.convert_text_to_embedding(
            model=self.embedding_model,
            text=query
        ).to('cpu')

        # Convert relations to embeddings and calculate similarity
        # TODO: Add text chunk in text before converting to embedding
        relation_embeddings = []
        for relation in relations:
            text = f"{relation['subject']} {relation['relation']} {relation['object']}"
            if self.embedding_model == "intfloat/multilingual-e5-small":
                text = "passage: " + text
            relation_embeddings.append(
                self.convert_text_to_embedding(
                    model=self.embedding_model,
                    text=text,
                ).to('cpu')
            )
        self.embedding_model.to('cpu')
        torch.cuda.empty_cache()

        # Calculate dot product between query and relation embeddings
        relation_embeddings_tensor = torch.stack(relation_embeddings)
        dot_products = query_embedding @ relation_embeddings_tensor.T

        # Add score to each relation based on the dot product
        for i, relation in enumerate(relations):
            relation['score'] = dot_products[i].item()

        # Get top-k relations based on dot product scores
        top_k_indices = torch.topk(dot_products, k=top_k).indices
        top_k_relations = [relations[i] for i in top_k_indices]

        return top_k_relations


if __name__ == "__main__":
    test = NHopRetriever(
        embedding_model="intfloat/multilingual-e5-small",
        NER_model="microsoft/phi-4"
    )
    a = test.retrieve(
        query="What is the LoRA?",
        n_hop=2,
        top_k=10
    )
    print(a)