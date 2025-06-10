"""
Splits legal documents into 1000-token chunks while preserving sentence boundaries using LlamaIndex.

Input: parsed_documents.json
Output: split_documents.json
"""


import json
import uuid
from llama_index.core.text_splitter import SentenceSplitter

def split_text_with_word_boundaries(text, max_length=1000, overlap=50):
    """
    Split text into chunks respecting word boundaries.
    
    Parameters:
    - text: The input text to split
    - max_length: Maximum tokens per chunk
    - overlap: Number of tokens to overlap between chunks
    
    Returns:
    - List of text chunks
    """
    splitter = SentenceSplitter(
        chunk_size=max_length,
        chunk_overlap=overlap,
        separator=" "
    )
    return splitter.split_text(text)


with open('parsed_documents.json', 'r', encoding='utf-8') as file:
    documents = json.load(file)

new_documents = []
chunk_counter = 1  

for doc in documents:
    text = doc['text'] if isinstance(doc['text'], str) else doc['text'][0]

    text_length = len(text)
    
    # Create a common ID for this document and its chunks
    common_id = str(uuid.uuid4())

    print(f"\nProcessing document: {common_id} | Original Length: {text_length}")

    # Only split texts longer than 3000 characters (roughly 1000 tokens)
    if text_length > 3000:
        print(f"🛠️ Splitting document with length {text_length}...")
        
        # Split text at word boundaries
        split_texts = split_text_with_word_boundaries(text, max_length=1000, overlap=50)
        print(f"Created {len(split_texts)} chunks.")

        for i, split_text in enumerate(split_texts):
            new_doc = doc.copy()
            new_doc['text'] = [split_text]
            new_doc['metadata'] = doc['metadata'].copy()
            
            # Add the common ID to link related chunks
            new_doc['metadata']['parent_id'] = common_id
            new_doc['metadata']['chunk_number'] = chunk_counter
            new_doc['metadata']['splitted'] = 'yes'
            
            new_documents.append(new_doc)
            chunk_counter += 1 
    else:
        print(f"Document {common_id} is not split (length ≤ 3000).")

        # For shorter documents, just add the ID
        doc_copy = doc.copy()
        doc_copy['metadata'] = doc['metadata'].copy()
        doc_copy['metadata']['parent_id'] = common_id
        doc_copy['metadata']['chunk_number'] = chunk_counter 
        doc_copy['metadata']['splitted'] = 'no'
        
        new_documents.append(doc_copy)
        chunk_counter += 1  

output_file = "large_documents.json"
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(new_documents, file, indent=2, ensure_ascii=False)

print(f"\n Output file: {output_file}")