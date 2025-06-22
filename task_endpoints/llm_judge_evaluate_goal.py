import asyncio
import logging
import os

from dotenv import load_dotenv

# Load OpenAI API client
from openai import AsyncOpenAI, OpenAI

from llm_interface.async_llm_inference import batch_generate

## import from local modules
from llm_interface.llm_inference import generate_with_openai
from task_configs.config import LLM_API_TIMEOUT, LLM_MAX_RETRIES, LLM_TASKS_CONFIG
from task_configs.prompt_prep import format_llm_judge_evaluate_goal_prompt

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Configure logging
logger = logging.getLogger(__name__)

# load environment variables
load_dotenv(override=True)


# Set OpenAI Client
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    organization=os.environ["OPENAI_ORG_ID"],
    timeout=LLM_API_TIMEOUT,
    max_retries=LLM_MAX_RETRIES,
)
# Set async client
async_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    organization=os.environ["OPENAI_ORG_ID"],
    timeout=LLM_API_TIMEOUT,
    max_retries=LLM_MAX_RETRIES,
)


def evaluate_single_goal(employee_data):
    # prepare the prompt for LLM to evaluate the quality of generated goals
    task_type = "llm_judge_evaluate_goal"
    llm_input_args_config = LLM_TASKS_CONFIG[task_type]["openai"]["llm_input_args"]
    # route to the correct function based on the given task type
    prompt_dict = format_llm_judge_evaluate_goal_prompt(
        task_type, employee_data, llm_input_args_config, goal=employee_data["goals"][0]
    )
    # generate the output using OpenAI client
    llm_output = generate_with_openai(client, prompt_dict, llm_input_args_config)

    return llm_output


async def process_single_employee_goals(employee_data, llm_input_args_config):
    """
    Process goals for a single employee asynchronously.
    """
    # Prepare the prompt for LLM to evaluate the quality of generated goals
    prompt_dicts = [
        format_llm_judge_evaluate_goal_prompt(
            goal, employee_data, llm_input_args_config
        )
        for goal in employee_data["goals"]
    ]

    # Generate outputs using async OpenAI client
    all_outputs = await batch_generate(
        async_client, prompt_dicts, llm_input_args_config
    )
    # remove metadata and missing_info from the output
    for output in all_outputs:
        output.pop("metadata", None)
        output.pop("missing_info", None)
    return all_outputs


def process_all_employee_goals(employees_data):
    task_type = "llm_judge_evaluate_goal"
    llm_input_args_config = LLM_TASKS_CONFIG[task_type]["openai"]["llm_input_args"]

    async def process_all_employees():
        tasks = [
            process_single_employee_goals(employee_data, llm_input_args_config)
            for employee_data in employees_data
        ]
        results = await asyncio.gather(*tasks)
        return results

    # Run the async function
    all_results = asyncio.run(process_all_employees())
    num_of_employees = len(all_results)
    num_of_goals = sum(len(employee["goals"]) for employee in employees_data)
    logger.info(
        f"Processed {num_of_goals} goals for {num_of_employees} employees asynchronously."
    )
    return all_results
