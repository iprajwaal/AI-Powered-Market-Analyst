# /Users/prajwal/Developer/AI-Powered-Market-Analyst/backend/src/llm/gemini.py

from typing import Dict
from typing import Any
from typing import List
from typing import Optional
from src.config.log_config import logger
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.generative_models import Part
from vertexai.preview.generative_models import GenerationConfig
from vertexai.preview.generative_models import HarmBlockThreshold
from vertexai.preview.generative_models import HarmCategory


def _create_generation_config() -> GenerationConfig:
    """
    Create a GenerationConfig object with the default values.

    Returns:
        GenerationConfig: The GenerationConfig object.
    """
    try:
        gen_config = GenerationConfig(
            temperature = 0.0,
            top_k = 1.0,
            candidate_count = 1,
            max_output_tokens = 8192,
            seed = 12345
        )
        return gen_config
    except Exception as e:
        logger.error(f"Error creating GenerationConfig object: {e}")
        raise

def _create_safety_settings() -> Dict[HarmCategory, HarmBlockThreshold]:
    """
    Create a dictionary of HarmCategory to HarmBlockThreshold objects with the default values.

    Returns:
        Dict[HarmCategory, HarmBlockThreshold]: The dictionary of HarmCategory to HarmBlockThreshold objects.
    """
    try:
        safety_settings = {
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
        }
        return safety_settings
    except Exception as e:
        logger.error(f"Error creating safety settings: {e}")
        raise


def generate(model: GenerativeModel, contents: List[Part]) -> Optional[str]:
    """
    Generate text using the given model and contents.

    Args:
        model (GenerativeModel): The model to use for text generation.
        contents (List[Part]): The contents to use for text generation.

    Returns:
        Optional[str]: The generated text, or None if an error occurred.
    """
    try:
        logger.info("Generating text from Gemini...")
        response = model.generate_content(
            contents=contents,
            generation_config=_create_generation_config(),
            safety_settings=_create_safety_settings()
        )

        if not response.text:
            logger.error("No text generated.")
            return None
        
        logger.info("Text generated successfully.")
        return response.text
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        return None
    
