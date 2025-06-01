


import os
LLM_BASE_URL = "http://localhost:11434"
GENERATOR_MODEL = os.getenv("GENERATOR_DEPLOYMENT_NAME")               

#==========================================================================================#



START_INDEX = 1

Questions= 9000

END_INDEX = 1

BATCH_SIZE = 10


#==========================================================================================#



Chunks= 10


output_file = f"questions_{Questions}.json"



OUTPUT_FOLDER = "11000Hyde"





# OUTPUT_FOLDER = f"RRRR_{Questions}_Questions/{Chunks}_K_Sparse"


# OUTPUT_FOLDER = f"RRRR_{Questions}_Questions/{Chunks}_K_Dense"


# OUTPUT_FOLDER = f"RRRR_{Questions}_Questions/{Chunks}_K_Hybrid"





# OUTPUT_FOLDER = f"Random_{Questions}_Questions/{Chunks}_K_Sparse"


# OUTPUT_FOLDER = f"Random_{Questions}_Questions/{Chunks}_K_Dense"

# OUTPUT_FOLDER = f"Random_{Questions}_Questions/{Chunks}_K_Hybrid










#==========================================================================================#



# USE_COSINE_FILTERING = True  
# COSINE_THRESHOLD = 0.5  



# ---------------SPARSE RETRIEVER------------#


# RETRIEVER_TYPE = "Sparse_Retriever"  
# SPARSE_K = Chunks
# HYDE_K = 0
# DENSE_K = 0 
# USE_HYDE = False 
# USE_LLM_FILTERING = False 
# USE_RERANKING = False       
# K_VALUES = [SPARSE_K] 
# RERANK_TOP_K = 30
# K_VALUES = [10, 15, 20] 

#------------------------------------------#



#---------------DENSE RETRIEVER----------#


RETRIEVER_TYPE = "Dense_Retriever"  
DENSE_K = Chunks
SPARSE_K = 0 
USE_HYDE = True 
K_VALUES = [ 10, 15, 20]
USE_LLM_FILTERING = False 
USE_RERANKING = False     
RERANK_TOP_K = 30


#--------------------------------------------#

#---------------Hybrid_Retriever------------#

# RETRIEVER_TYPE = "Hybrid_Retriever"        
# TOP_K = Chunks
# ALPHA = 0.7
# SPARSE_K = 0                       
# HYDE_K = 0                       
# DENSE_K = 0                         
# USE_LLM_FILTERING = False 
# USE_RERANKING = False      
# RERANK_TOP_K = 30              
# K_VALUES = [5, 10, 15, 20]  
# USE_HYDE = False    



# RETRIEVER_TYPE = "Hybrid_Retriever"    
# LLAMA_HYBRID_ALPHA = 0.8        
# LLAMA_HYBRID_CANDIDATE_K = 30    
# USE_HYDE = False               
# SPARSE_K = 0               
# HYDE_K = 0                       
# DENSE_K = 0                       
# USE_LLM_FILTERING = False         
# USE_RERANKING = False             
# RERANK_TOP_K = 30                 
# K_VALUES = [10, 15, 20]        

#------------------------------------------#




# #---------------HYDE RETRIEVER----------#


# RETRIEVER_TYPE = "dense_azure"  
# DENSE_K = 20  # Number of dense results
# USE_HYDE = True 
# K_VALUES = [DENSE_K]
# SPARSE_K = 0
# USE_LLM_FILTERING = False  
# USE_RERANKING = False  
# # RERANK_TOP_K = 50

# --------------------------------------------


K_VALUE = K_VALUES[0]






CHUNKS_FILE = "document_embeddings/split_documents_deleted.json"  



# CHUNKS_FILE = "FINAL_processed_documents_cleaned.json"  















































# EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
# GENERATOR_DEPLOYMENT_NAME = os.getenv("GENERATOR_DEPLOYMENT_NAME")

     
# JSON_HISTORY_FILE = "chat_history.json"



