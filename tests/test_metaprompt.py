import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.metaprompt import MetaPrompt

@pytest.fixture
def meta_prompt_instance():
    return MetaPrompt()

def test_generate_prompt(meta_prompt_instance):
    task = "Translate English to French"
    variables = "text"
    prompt, extracted_vars = meta_prompt_instance(task, variables)
    assert isinstance(prompt, str)
    assert prompt.strip() != ""
    assert "{{TEXT}}" in prompt
    assert "TEXT" in extracted_vars
