import spacy
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Add the parent directory to the path to allow imports from task1
sys.path.append(str(Path(__file__).resolve().parent.parent))

from task1.process_data.labeling import load_spans
from task1.process_data.utils import load_article_text, resolve_article_path

def create_sentence_labels(articles_dir: Path, labels_file: Path, output_file: Path):
    """
    Creates a TSV file with sentence-level labels by reusing existing functions.
    """
    # Load spacy model
    nlp = spacy.load("en_core_web_sm")

    # 1. Reuse `load_spans` to read propaganda spans
    propaganda_spans = load_spans(labels_file)

    # Process articles
    sentence_data = []
    
    # Get a list of all article IDs from the keys of the propaganda_spans dictionary
    article_ids = sorted(propaganda_spans.keys())

    for article_id in tqdm(article_ids, desc=f"Processing sentences in {articles_dir.name}"):
        try:
            # 2. Reuse `resolve_article_path` and `load_article_text`
            article_path = resolve_article_path(article_id, articles_dir)
            text = load_article_text(article_path)
        except FileNotFoundError:
            print(f"Warning: Article {article_id} not found in {articles_dir}. Skipping.", file=sys.stderr)
            continue

        doc = nlp(text)
        article_propaganda_spans = propaganda_spans.get(article_id, [])

        for sent in doc.sents:
            sent_start = sent.start_char
            sent_end = sent.end_char
            sent_text = sent.text.strip()

            if not sent_text:
                continue

            has_propaganda = 0
            for prop_start, prop_end in article_propaganda_spans:
                if max(sent_start, prop_start) < min(sent_end, prop_end):
                    has_propaganda = 1
                    break
            
            sentence_data.append({
                "article_id": article_id,
                "sentence": sent_text,
                "label": has_propaganda
            })

    # Save to TSV
    df = pd.DataFrame(sentence_data)
    output_file.parent.mkdir(exist_ok=True)
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Successfully created {output_file}")

