import os, shutil
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader
import yaml
from pyprojroot import here
import nltk
from rake_nltk import Rake
import requests




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

def get_Lebron():
    
    #first delete whatever is in that directory already
    data_dir = './src/data'

    # Clear the contents of the directory before saving the new file
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Removes files and symlinks
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Removes directories
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    
    # URL of the new LeBron webpage to download
    
    
    url = 'https://www.basketball-reference.com/players/j/jamesle01.html'

    response = requests.get(url)
    response.raise_for_status()

    
    file_path = './src/data/lebron_james_reference.html'


    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(response.text)
    
    print('New LeBron File Downloaded')

    return 


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