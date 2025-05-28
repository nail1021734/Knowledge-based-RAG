import argparse
import os
from typing import Dict, List

from tqdm import tqdm

from neo4j import GraphDatabase
from utils._path import DATA_SOURCE, DATABASE
from utils.json_tools import load_json, save_json


def connect_to_neo4j() -> GraphDatabase.driver:
    """Connect to the Neo4j database.
    Returns:
        GraphDatabase.driver: A driver instance for connecting to the Neo4j database.
    """
    # Replace with your Neo4j connection details
    # Ensure you have the Neo4j Python driver installed: pip install neo4j
    URI = "neo4j://localhost:7687"
    USERNAME = "neo4j"
    PASSWORD = "password"
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    return driver


def import_transactions(data: List[Dict], driver: GraphDatabase.driver, database_name: str) -> None:
    """Import triplets into the Neo4j database.
    Args:
        triplets (List[Dict]): A list of triplets to import.
        driver (GraphDatabase.driver): The Neo4j driver instance.
    """
    with driver.session() as session:
        for chunk in tqdm(data, desc="Importing triplets to Neo4j"):
            for triplet in chunk['triplets']:
                # Create nodes for subject and object
                session.run(
                    """
                    MERGE (s: ENTITY {name: $subject})
                    MERGE (p: ENTITY {name: $object})
                    MERGE (s)-[r: RELATION {name: $relation}]->(p)
                    SET r.database = $database_dir,
                        r.text_chunk_index = $text_chunk_index
                    """,
                    subject=triplet[0],
                    relation=triplet[1],
                    object=triplet[2],
                    database_dir=database_name,
                    text_chunk_index=chunk['text_chunk_index']
                )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Neo4j Database Builder")
    parser.add_argument(
        "--database_dir",
        type=str,
        help="Specify the directory to import the database files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    full_database_dir = os.path.join(DATABASE, args.database_dir)

    # Load the triplets from the JSON file
    triplets_path = os.path.join(full_database_dir, 'cleaned_triplets.json')
    triplets_data = load_json(triplets_path)

    # Connect to Neo4j
    driver = connect_to_neo4j()

    # Import triplets into Neo4j
    import_transactions(triplets_data, driver, args.database_dir)

    # Close the Neo4j driver connection
    driver.close()