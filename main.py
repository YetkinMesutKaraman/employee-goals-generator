import os

import pandas as pd

from scripts.generate_company_data import (
    generate_employee_data,
    write_to_csv,
    write_to_json,
)
from task_configs.config import LLM_TASKS_CONFIG
from task_endpoints.generate_employee_goals import generate_batch_employee_goals
from task_endpoints.llm_judge_evaluate_goal import process_all_employee_goals

# NUMBER OF EMPLOYEES
NUM_EMPLOYEES = 50
# PROVİDER and MODEL
PROVIDER = "openai"
MODEL = "gpt-4.1-nano"
# Output directory
OUTPUT_DIR = "output_data"
# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
# File path for the generated data
FILE_NAME = f"synthetic_employee_data_{NUM_EMPLOYEES}_{PROVIDER}_{MODEL}"


def main():
    # generate company data
    df_employee = generate_employee_data()
    # save to CSV
    write_to_csv(df_employee, OUTPUT_DIR, FILE_NAME)
    # read the employee data from saved CSV
    df_employee = pd.read_csv(OUTPUT_DIR + "/" + FILE_NAME + ".csv")
    print(f"Loaded {len(df_employee)} employee records from CSV.")
    # print the first few records
    print(df_employee.head())
    # generale employee goals all at once
    all_employee_goals = generate_batch_employee_goals(
        df_employee,
        LLM_TASKS_CONFIG["generate_employee_goals"]["openai"]["llm_input_args"],
    )
    # print the first few generated goals
    print(f"Generated Goals for Employees: {len(all_employee_goals)}")
    print(all_employee_goals[:5])  # Print first 5 goals for brevity
    # add the generated goals to the DataFrame
    # parse just goals from generate_batch_employee_goals
    goals_list = [
        single_employee_goals["goals"] for single_employee_goals in all_employee_goals
    ]
    df_employee["goals"] = goals_list
    print("First 5 employee goals:")
    print(df_employee.head())
    # save the updated DataFrame with goals to json
    write_to_json(df_employee, OUTPUT_DIR, FILE_NAME + "_with_goals")
    print(
        f"\nGenerated {len(df_employee)} employee records with goals and saved to {OUTPUT_DIR}/{FILE_NAME}_with_goals.json"
    )
    # llm judge evaluate goals

    all_evaluated_goals = process_all_employee_goals(
        df_employee.to_dict(orient="records")
    )
    # print the first few evaluated goals
    print("Evaluated Goals for Employees:")
    print(all_evaluated_goals[:1])  # Print first 5 evaluations for brevity
    # add the evaluated goals to the DataFrame

    df_employee["evaluated_goals"] = all_evaluated_goals

    print("\nFirst employee's details with evaluated goals from DataFrame:")
    # Print relevant columns for the first employee to check
    # Using to_string() for better console output of complex cell content
    print(df_employee.head(1).to_string())

    # save the updated DataFrame with evaluated goals to json
    output_filename_evaluated = FILE_NAME + "_with_evaluated_goals"
    write_to_json(df_employee, OUTPUT_DIR, output_filename_evaluated)
    print(
        f"\nUpdated DataFrame with evaluated goals and saved to "
        f"{OUTPUT_DIR}/{output_filename_evaluated}.json"
    )


if __name__ == "__main__":
    main()
