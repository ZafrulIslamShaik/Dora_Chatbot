
from langchain_openai import AzureChatOpenAI # warning solved 
from langchain_ollama import OllamaLLM  
from EVALUATION.config import *
from dotenv import load_dotenv
import os 

load_dotenv(".env.local")

def get_azure_llm():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_deployment=os.getenv("GENERATOR_DEPLOYMENT_NAME"),
        api_version="2023-05-15",
        temperature=0.0,
        max_tokens=10000
    )
