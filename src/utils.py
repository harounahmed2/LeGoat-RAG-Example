import os, shutil
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader
import yaml
from pyprojroot import here
import nltk
from rake_nltk import Rake




def RAG(_config, _docs):
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(
            model=_config.gpt_model,
            temperature=_config.temperature,
            max_tokens=_config.max_tokens,
            system_prompt=_config.llm_system_role,
        ),
        chunk_size=_config.chunk_size,
    )
    index = VectorStoreIndex.from_documents(_docs, service_context=service_context)
    return index

def load_data():
    reader = SimpleDirectoryReader(input_dir="src/data", recursive=True)
    docs = reader.load_data()
    return docs

def parse_query(query):
    rake = Rake()
    rake.extract_keywords_from_text(query)
    keywords = rake.get_ranked_phrases()
    return " ".join(keywords)





class LoadConfig:
    """
    A class for loading configuration settings, including OpenAI credentials.

    This class reads configuration parameters from a YAML file and sets them as attributes.
    It also includes a method to load OpenAI API credentials.

    Attributes:
        gpt_model (str): The GPT model to be used.
        temperature (float): The temperature parameter for generating responses.
        llm_system_role (str): The system role for the language model.
        llm_format_output (str): The formatting constrain of the language model.

    Methods:
        __init__(): Initializes the LoadConfig instance by loading configuration from a YAML file.
        load_openai_credentials(): Loads OpenAI configuration settings.
    """

    def __init__(self) -> None:
        with open(here("config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)
        self.gpt_model = app_config["gpt_model"]
        self.temperature = app_config["temperature"]
        self.max_tokens = app_config["max_tokens"]
        self.llm_system_role = app_config["llm_system_role"]
        self.llm_format_output = app_config["llm_format_output"]
        self.chunk_size = app_config["chunk_size"]
        self.similarity_top_k = app_config["similarity_top_k"]