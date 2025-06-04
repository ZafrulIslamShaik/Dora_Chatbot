


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

OUTPUT_FOLDER = "Test"



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


CHUNKS_FILE = "document_embeddings/split_documents.json"  




















