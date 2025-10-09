import os
import pymupdf
import arxiv
import requests
import json
import re
import time

# 1. Title extraction from arXiv API
def extract_title_from_arxiv(filename):
    arxiv_pattern = r"(\d{4}\.\d{4,5})(v\d+)?"
    match = re.match(arxiv_pattern,filename.replace('.pdf',''))

    if match:
        arxiv_id = match.group(1)
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results())
            return paper.title,'arxiv'
        except Exception as e:
            print(f"arxiv fetch failed : {e}")

    return None,None

# 2. Title extraction from pdf metadata
def extract_title_from_pdf_metadata(pdf_path):
    try:
        doc = pymupdf.open(pdf_path)
        metadata = doc.metadata
        doc.close()

        if metadata and metadata.get('title'):
            title = metadata['title'].strip()
            if len(title)>10 and len(title)<200:
                return title, 'pdf_metadata'
    except:
        pass
    return None,None

def extract_title_from_pdf_text(pdf_path):
    """Enhanced title extraction from PDF text"""
    try:
        doc = pymupdf.open(pdf_path)
        first_page = doc[0]
        
        # Get all text blocks with formatting
        blocks = first_page.get_text("dict", sort=True)["blocks"]
        
        candidates = []
        
        for block in blocks[:15]:  # Check more blocks
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    size = span["size"]
                    
                    # Filter criteria for potential titles
                    if (len(text) > 15 and len(text) < 200 and 
                        not text.isnumeric() and
                        not text.startswith('http') and
                        not re.match(r'^\d+$', text) and
                        not text.lower().startswith('abstract')):
                        
                        candidates.append({
                            'text': text,
                            'size': size,
                            'y_position': span['origin'][1]  # Vertical position
                        })
        
        doc.close()
        
        if not candidates:
            return None, None
        
        # Sort by font size (descending) and y-position (ascending = higher on page)
        candidates.sort(key=lambda x: (-x['size'], x['y_position']))
        
        # Get the largest text that's near the top
        if candidates:
            title = candidates[0]['text'].replace('\n', ' ').strip()
            return title, 'pdf_text'
            
    except Exception as e:
        print(f"    PDF text extraction error: {e}")
    
    return None, None

def search_semantic_scholar(title_guess):
    """Try to find paper on Semantic Scholar to verify/get proper title"""
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': title_guess,
            'limit': 1,
            'fields': 'title'
        }
        
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('data') and len(data['data']) > 0:
                return data['data'][0]['title'], 'semantic_scholar'
    except:
        pass
    return None, None

def clean_filename_as_title(filename):
    """Convert filename to readable title as last resort"""
    # Remove extension and numbers
    name = filename.replace('.pdf', '')
    name = re.sub(r'^\d+_', '', name)  # Remove leading numbers
    name = re.sub(r'^\d{4}\.\d{4,5}(v\d+)?', '', name)  # Remove arXiv ID
    
    # Replace underscores and clean up
    name = name.replace('_', ' ').strip()
    
    # Title case
    if name:
        return ' '.join(word.capitalize() for word in name.split()), 'filename'
    
    return filename.replace('.pdf', ''), 'filename'



def get_paper_title_multi_strategy(pdf_path,filename):

    """
    Try multiple strategies in order of reliability:
    1. arXiv API (most reliable)
    2. PDF metadata
    3. PDF text extraction
    4. Semantic Scholar verification
    5. Cleaned filename (fallback)
    """
    print(f"  Trying arXiv API...", end='')
    title, source = extract_title_from_arxiv(filename)
    if title:
        print(f" ✓ Found!")
        return title,source
    print(" ✗")
    
    print(f"Trying pdf metadata...",end='')
    title,source = extract_title_from_pdf_metadata(pdf_path)
    if title:
        print(f" ✓ Found!")
        return title,source
    print(" ✗")

    print(f"  Trying PDF text extraction...", end='')
    title, source = extract_title_from_pdf_text(pdf_path)
    if title:
        print(f" ✓ Found!")
        
        # Optional: Verify with Semantic Scholar
        print(f"  Verifying with Semantic Scholar...", end='')
        verified_title, verified_source = search_semantic_scholar(title)
        if verified_title:
            print(f" ✓ Verified!")
            return verified_title, 'semantic_scholar'
        print(" ✗")
        
        return title, source
    print(" ✗")
    
    print(f"  Using filename...", end='')
    title, source = clean_filename_as_title(filename)
    print(f" ✓")
    return title, source

if __name__ == "__main__":
    papers_folder = "Papers"
    pdf_files = [f for f in os.listdir(papers_folder) if f.endswith(".pdf")]
    print(f"Extracting titles from {len(pdf_files)} papers...\n")
    print("="*70)

    extracted_titles = {}
    sources_count = {
        'arxiv':0,
        'pdf_metadata':0,
        'pdf_text': 0,
        'semantic_scholar': 0,
        'filename': 0
    }

    for i, filename in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] {filename}")
        path = os.path.join(papers_folder, filename)
        
        title, source = get_paper_title_multi_strategy(path, filename)
        
        extracted_titles[filename] = {
            'title': title,
            'source': source,
            'needs_manual_review': source == 'filename'  # Flag for manual check
        }
        
        sources_count[source] += 1
        
        print(f"  → Title: {title[:70]}{'...' if len(title) > 70 else ''}")
        print(f"  → Source: {source}")
        
        # Rate limiting for API calls
        if source in ['arxiv', 'semantic_scholar']:
            time.sleep(0.5)

    with open('paper_titles.json', 'w', encoding='utf-8') as f:
        json.dump(extracted_titles, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for source, count in sources_count.items():
        print(f"{source:20s}: {count:3d} papers ({count/len(pdf_files)*100:.1f}%)")
    
    needs_review = sum(1 for v in extracted_titles.values() if v['needs_manual_review'])
    print(f"\n⚠ {needs_review} papers need manual review (check paper_titles.json)")
    print(f"✓ Results saved to 'paper_titles.json'")