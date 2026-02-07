from __future__ import annotations

import spacy
from spacy.language import Language


@Language.component("newline_boundary")
def _newline_boundary_component(doc):
    """Custom spaCy component to force sentence segments on newlines."""
    for i, token in enumerate(doc[:-1]):
        if "\n" in token.text:
            doc[i + 1].is_sent_start = True
    return doc


def get_configured_spacy(model_name: str = "en_core_web_sm"):
    """Load and configure spaCy model with custom boundary rules."""
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(
            "Falling back to blank English spaCy model; install %s for best accuracy",
            model_name,
        )
        nlp = spacy.blank("en")

    if (
        not nlp.has_pipe("parser")
        and not nlp.has_pipe("senter")
        and "sentencizer" not in nlp.pipe_names
    ):
        nlp.add_pipe("sentencizer")

    if "newline_boundary" not in nlp.pipe_names:
        nlp.add_pipe("newline_boundary", before="parser" if nlp.has_pipe("parser") else None)

    return nlp
