import json

# Load the JSON file
with open('split_documents.json', 'r') as file:
    data = json.load(file)

# Dictionary to group by parent_id
grouped_by_parent = {}

# Process each item
for item in data:
    metadata = item['metadata']
    parent_id = metadata['parent_id']
    references = metadata['references']
    
    # Clean references - remove 'Point' field
    cleaned_references = []
    for ref in references:
        cleaned_ref = {
            'document': ref.get('document', ''),
            'article_number': ref.get('article_number', []),
            'Paragraph': ref.get('Paragraph', [])
        }
        cleaned_references.append(cleaned_ref)
    
    # Add to grouped dictionary
    if parent_id not in grouped_by_parent:
        grouped_by_parent[parent_id] = []
    
    grouped_by_parent[parent_id].extend(cleaned_references)

# Deduplicate references within each parent_id group
for parent_id in grouped_by_parent:
    unique_references = []
    seen_references = set()
    
    for ref in grouped_by_parent[parent_id]:
        # Create a tuple for comparison (to check duplicates)
        ref_tuple = (ref['document'], str(ref['article_number']), str(ref['Paragraph']))
        
        if ref_tuple not in seen_references:
            seen_references.add(ref_tuple)
            unique_references.append(ref)
    
    grouped_by_parent[parent_id] = unique_references

# Create final output format
output = []
for parent_id, references in grouped_by_parent.items():
    output.append({
        'parent_id': parent_id,
        'references': references
    })

# Save to new JSON file
with open('output.json', 'w') as file:
    json.dump(output, file, indent=2)

print("Done!")