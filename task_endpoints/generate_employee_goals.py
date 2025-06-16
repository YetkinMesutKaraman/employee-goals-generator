import asyncio
import logging
import os
from pprint import pprint
from time import perf_counter

from dotenv import load_dotenv

# Load OpenAI API client
from openai import AsyncOpenAI, OpenAI

from llm_interface.async_llm_inference import batch_generate
from llm_interface.llm_inference import generate_with_openai
from task_configs.config import LLM_API_TIMEOUT, LLM_MAX_RETRIES, LLM_TASKS_CONFIG
from task_configs.prompt_prep import route_prompt_formatting

## import from local modules
from utils.data_prep import load_employee_data

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


def evaluate_goal_quality(goals: list) -> dict:
    def is_specific(goal):
        return any(char.isdigit() for char in goal) or any(
            word in goal.lower()
            for word in ["by", "within", "%", "$", "increase", "reduce"]
        )

    def is_measurable(goal):
        return any(
            word in goal.lower()
            for word in ["kpi", "metric", "target", "percent", "score", "OKR"]
        )

    vague_count = sum(1 for g in goals if not is_specific(g))
    measurable_count = sum(1 for g in goals if is_measurable(g))

    return {
        "total_goals": len(goals),
        "vague_goals": vague_count,
        "measurable_goals": measurable_count,
        "specific_goals": len(goals) - vague_count,
        "clarity_score": round((len(goals) - vague_count) / len(goals), 2)
        if goals
        else 0.0,
        "measurability_score": round(measurable_count / len(goals), 2)
        if goals
        else 0.0,
    }


# generate goals for a single employee
def generate_single_employee_goals(employee_data: dict, llm_input_args_config: dict):
    """
    Generate goals for a single employee using OpenAI client.

    Args:
        employee_data (dict): Dictionary containing employee data.
        llm_input_args_config (dict): Configuration for LLM input arguments.

    Returns:
        dict: Generated goals and metadata.
    """
    task_type = "generate_employee_goals"
    # route to the correct function based on the given task type
    prompt_dict = route_prompt_formatting(
        task_type, employee_data, llm_input_args_config
    )
    # generate the output using OpenAI client
    llm_output = generate_with_openai(client, prompt_dict, llm_input_args_config)
    return llm_output


# generate goals for a batch of employees
def generate_batch_employee_goals(df_employee, llm_input_args_config: dict) -> list:
    """
    Generate goals for a batch of employees using OpenAI client.

    Args:
        df_employee (DataFrame): DataFrame containing employee data.
        llm_input_args_config (dict): Configuration for LLM input arguments.

    Returns:
        list: List of generated goals for each employee.
    """
    task_type = "generate_employee_goals"
    # prepare batch data for processing
    employee_dict_list = df_employee.to_dict(orient="records")
    # prepare batch data for LLM input
    formatted_prompt_dict_list = [
        route_prompt_formatting(task_type, employee_data, llm_input_args_config)
        for employee_data in employee_dict_list
    ]
    # Run async goal generation
    all_outputs = asyncio.run(
        batch_generate(
            async_client,
            formatted_prompt_dict_list,
            llm_input_args_config,
        )
    )
    return all_outputs


def main():
    task_type = "generate_employee_goals"
    llm_input_args_config = LLM_TASKS_CONFIG[task_type]["openai"]["llm_input_args"]
    is_single_input = False  # Set to True for single employee input
    if is_single_input:
        # Example single employee input
        employee_data = {
            "name": "Ava Liu",
            "job_title": "Software Engineer",
            "team_function": "Engineering",
        }
        # Generate goals for a single employee
        llm_output = generate_single_employee_goals(
            employee_data, llm_input_args_config
        )
        # Print the output
        print("Generated Goals:")
        pprint(llm_output)
    else:
        # Example batch processing with a DataFrame
        df_employee = load_employee_data(
            "./output_data/synthetic_employee_data_50_openai_gpt-4.1-nano.csv"
        )

        # Run async goal generation
        start_time = perf_counter()
        print("\nStarting batch goal generation...")
        all_outputs = generate_batch_employee_goals(df_employee, llm_input_args_config)
        end_time = perf_counter()
        print(f"Batch goal generation completed in {end_time - start_time:.2f} seconds")
        print("\nGenerated Goals for all employees:")
        pprint(all_outputs)


# Example Usage
if __name__ == "__main__":
    main()
