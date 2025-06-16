# This module provides functionality to generate goals using an OpenAI client.
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Configure logging
logger = logging.getLogger(__name__)


def prepare_llm_input_args(
    llm_input_args_config: dict,
    prompt_dict: dict,
) -> dict:
    # Create a copy to avoid mutating the original config
    llm_input_args = llm_input_args_config.copy()

    # Add the messages to the configuration
    llm_input_args["input"] = [
        {
            "role": "system",
            "content": prompt_dict["system_prompt"],
        },
        {
            "role": "user",
            "content": prompt_dict["user_prompt"],
        },
    ]

    # Remove system_message and user_message from the final args since it's now in messages
    llm_input_args.pop("system_prompt", None)
    llm_input_args.pop("user_prompt", None)

    return llm_input_args


def add_metadata_to_llm_output(llm_output: dict, prompt_dict: dict) -> dict:
    """
    Add metadata and missing information to the LLM output.
    Args:
        llm_output (dict): The output from the LLM.
        prompt_dict (dict): Dictionary containing metadata and missing information.
    Returns:
        dict: Updated output with metadata and missing information.
    """
    llm_output["metadata"] = prompt_dict.get("metadata", {})
    llm_output["missing_info"] = prompt_dict.get("missing_info", [])
    return llm_output


def generate_with_openai(
    client, prompt_dict: dict, llm_input_args_config: dict
) -> dict:
    """
    Generate goals using OpenAI client based on the provided prompt and configuration.

    Args:
        client: OpenAI client instance.
        prompt_dict (dict): Dictionary containing system and user messages.
        llm_input_args_config (dict): Configuration for LLM input arguments.

    Returns:
        dict: Generated goals and metadata.
    """
    llm_input_args = prepare_llm_input_args(llm_input_args_config, prompt_dict)

    response = client.responses.parse(**llm_input_args)

    json_output = response.output_text
    validated_output = (
        llm_input_args_config["text_format"]
        .model_validate_json(json_output)
        .model_dump()
    )
    logger.info("LLM output validated successfully.")
    # add metadata to the output
    validated_output = add_metadata_to_llm_output(validated_output, prompt_dict)

    return validated_output
