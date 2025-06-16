## Our task space: ["generate_employee_goals", "llm_judge_evaluate_goals"]


from task_configs.schemas import EmployeeGoals, GoalEvaluation

LLM_API_TIMEOUT = 300  # seconds
LLM_MAX_RETRIES = 0  # Number of retries for API calls

LLM_TASKS_CONFIG = {
    "generate_employee_goals": {
        "openai": {
            "llm_input_args": {
                "model": "gpt-4.1-nano",
                "system_prompt": """
You are a helpful HR assistant designed to help employees set high-quality performance goals for the next 3–6 months.

You will generate concise, measurable, role-appropriate goals based on the employee’s job context.

Goals must be:
- Specific and measurable
- Aligned with the employee’s role, level, and org/team priorities
- Clear and forward-looking

Return 3–5 goals using bullet points. Avoid vague or generic phrasing.
""".strip(),
                "timeout": LLM_API_TIMEOUT,  # seconds
                "max_output_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.98,
                "user_prompt": """
The following is the context for an employee:

- Name: {name}
- Job Title: {job_title}
- Seniority Level: {seniority_level}
- Team/Function: {team_function}
- Manager/Org Priorities: {manager_org_priorities}

Generate 3–5 high-quality performance goals for the next 3–6 months. 
The goals should be specific, measurable, and tailored to the employee’s role, seniority, and organizational context.

If the employee is in a managerial or leadership role, include aspects like team impact, delivery strategy, and people development.
If the employee is an individual contributor, focus on personal execution, delivery, and growth within their role.
""".strip(),
                "text_format": EmployeeGoals,
            },
        },
        "Retry": {
            "max_retries": 3,
        },
    },
    "llm_judge_evaluate_goal": {
        "openai": {
            "llm_input_args": {
                "model": "gpt-4.1-mini",
                "system_prompt": """You are an expert evaluator of employee performance goals. You assess goals based on clarity, specificity, role appropriateness, and measurability. Your evaluations are concise and reliable.""".strip(),
                "timeout": LLM_API_TIMEOUT,  # seconds
                "max_output_tokens": 300,
                "temperature": 0.1,
                "top_p": 0.98,
                "user_prompt": """
Evaluate the following goal across 4 dimensions:

- Clarity: Is the goal understandable and free of jargon or vagueness?
- Specificity: Does the goal avoid generalities and describe what should be achieved?
- Role Fit: Is the goal appropriate for the employee’s role and seniority?
- Measurability: Can the success of the goal be measured or tracked objectively?

Use the following scoring:
- Clarity, Specificity: Low / Medium / High
- Role Fit, Measurability: No / Somewhat / Yes

Return your evaluation for each dimension.

Goal:
"{goal}"

Job Title: {job_title}
Seniority Level: {seniority_level}
Team: {team_function}
""".strip(),
                "text_format": GoalEvaluation,
            },
        },
        "Retry": {
            "max_retries": 3,
        },
    },
}
