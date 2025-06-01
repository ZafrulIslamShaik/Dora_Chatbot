
from langchain_openai import AzureChatOpenAI # warning solved 
from langchain_ollama import OllamaLLM  
from config import *
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

def get_ollama_llm():
    return OllamaLLM(model=GENERATOR_MODEL, base_url=LLM_BASE_URL)

def get_llm(llm_type="azure"):
    if llm_type == "azure":
        return get_azure_llm()
    return get_ollama_llm()