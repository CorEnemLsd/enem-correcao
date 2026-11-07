"""Text preprocessing for UOL essay corpus."""
import re


def clean_uol_text(text: str) -> str:
    """Remove UOL correction markers and clean text."""
    lines = text.split('\n')
    cleaned_lines = []
    skip_next = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip UOL markers completely
        if line.startswith('Redação corrigida'):
            continue
        if line.startswith('Inconsistente Erro Correção'):
            skip_next = True  # Skip the next lines with corrections
            continue
        if skip_next:
            # Skip correction marker lines (typically repeated words)
            continue
            
        # Extract content after "Tema:" - this is the prompt, not the essay
        if line.startswith('Tema:'):
            continue  # Don't include tema in essay text
            
        # Extract content after "Texto:"
        if line.startswith('Texto:'):
            line = line[6:].strip()  # Remove "Texto:"
            
        cleaned_lines.append(line)
    
    # Join and deduplicate CONSECUTIVE words only
    text = ' '.join(cleaned_lines)
    
    # Deduplicate consecutive words (case-insensitive) - only immediate repeats
    words = text.split()
    deduped = []
    prev_lower = None
    for word in words:
        if word.lower() != prev_lower:
            deduped.append(word)
            prev_lower = word.lower()
    
    return ' '.join(deduped)


def clean_essay_text(text: str) -> str:
    """Generic essay text cleaning."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
