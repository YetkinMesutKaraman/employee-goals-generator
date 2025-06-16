import asyncio
import logging

from llm_interface.llm_inference import (
    add_metadata_to_llm_output,
    prepare_llm_input_args,
)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Configure logging
logger = logging.getLogger(__name__)


async def generate_with_openai_async(
    async_client, formatted_prompt_dict: dict, llm_input_args_config: dict
) -> dict:
    """
    Generate goals using OpenAI client based on the provided prompt and configuration.

    Args:
        client: OpenAI client instance.
        formatted_prompt_dict (dict): Dictionary containing system and user messages.
        llm_input_args_config (dict): Configuration for LLM input arguments.

    Returns:
        dict: Generated goals and metadata.
    """
    llm_input_args = prepare_llm_input_args(
        llm_input_args_config, formatted_prompt_dict
    )

    response = await async_client.responses.parse(**llm_input_args)

    json_output = response.output_text
    validated_output = (
        llm_input_args_config["text_format"]
        .model_validate_json(json_output)
        .model_dump()
    )
    logger.info("LLM output validated successfully.")

    # add metadata to the output
    validated_output = add_metadata_to_llm_output(
        validated_output, formatted_prompt_dict
    )

    return validated_output


# batch processing for generating goals using asyncio gather
async def batch_generate(
    async_client,
    formatted_prompt_dict_list: list[dict],
    llm_input_args_config: dict,
):
    tasks = [
        generate_with_openai_async(async_client, prompt_dict, llm_input_args_config)
        for prompt_dict in formatted_prompt_dict_list
    ]

    results = await asyncio.gather(*tasks)
    return results
