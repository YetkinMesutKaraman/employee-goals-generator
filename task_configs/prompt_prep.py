import pandas as pd


# detect missing data, return missing keys
def detect_missing_info(employee: dict) -> list:
    required_keys = [
        "name",
        "job_title",
        "seniority_level",
        "team_function",
        "manager_org_priorities",
    ]
    missing_keys = []
    for key in required_keys:
        if (
            key not in employee
            or pd.isna(employee[key])
            or (isinstance(employee[key], str) and not employee[key].strip())
        ):
            missing_keys.append(key)

    return missing_keys


def format_goal_generation_prompt(
    employee_context: dict, llm_input_args_config: dict
) -> dict:
    """
    Format the prompt for LLM to generate employee performance goals.
    Args:
        employee_context (dict): Dictionary containing employee information.
        llm_input_args_config (dict): Configuration for LLM input arguments.
    Returns:
        dict: Formatted prompt dictionary.
    """
    name = employee_context.get("name", "Unknown")
    job_title = employee_context.get("job_title", "Unknown")
    seniority_level = employee_context.get("seniority_level", "Unknown")
    team_function = employee_context.get("team_function", "Unknown")
    manager_org_priorities = employee_context.get(
        "manager_org_priorities", "Not provided"
    )

    user_prompt = llm_input_args_config["user_prompt"].format(
        name=name,
        job_title=job_title,
        seniority_level=seniority_level,
        team_function=team_function,
        manager_org_priorities=manager_org_priorities,
    )
    missing_keys = detect_missing_info(employee_context)

    return {
        "system_prompt": llm_input_args_config["system_prompt"],
        "user_prompt": user_prompt,
        "metadata": {
            "employee_name": name,
            "job_title": job_title,
        },
        "missing_info": missing_keys,
    }


def format_llm_judge_evaluate_goal_prompt(
    goal: str, employee_context: dict, llm_input_args_config: dict
) -> dict:
    """
    Format the prompt for LLM to evaluate the quality of generated goals.

    Args:
        goal (str): The goal to be evaluated.
        employee_context (dict): Dictionary containing employee information.
        llm_input_args_config (dict): Configuration for LLM input arguments.

    Returns:
        dict: Formatted prompt dictionary.
    """
    name = employee_context.get("name", "Unknown")
    job_title = employee_context.get("job_title", "Unknown")
    seniority_level = employee_context.get("seniority_level", "Unknown")
    team_function = employee_context.get("team_function", "Unknown")

    user_prompt = llm_input_args_config["user_prompt"].format(
        goal=goal,
        name=name,
        job_title=job_title,
        seniority_level=seniority_level,
        team_function=team_function,
    )
    missing_keys = detect_missing_info(employee_context)

    return {
        "system_prompt": llm_input_args_config["system_prompt"],
        "user_prompt": user_prompt,
        "metadata": {
            "employee_name": name,
            "job_title": job_title,
            "goal": goal,
        },
        "missing_info": missing_keys,
    }
