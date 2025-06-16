# Define the mock data schema for a synthetic company
import os
import uuid

from dotenv import load_dotenv
from mostlyai import mock

# load environment variables
load_dotenv(override=True)

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


# Compose a prompt for diverse roles and realistic context
prompt = (
    "Employees of a synthetic tech company. "
    "Diverse roles (ICs and managers) across Engineering, Product, Design, GTM, and Ops. "
    "Each entry should have: Name, Job Title, Seniority Level (e.g., Junior, Mid, Senior, Lead, Director, VP), "
    "Team/Function, and Manager/Organization Priorities (short text). "
    "Some entries should have incomplete or weak info (e.g., missing priorities, vague job titles, or missing seniority) to test robustness."
)

columns = {
    "name": {"prompt": "Full name of the employee", "dtype": "string"},
    "job_title": {
        "prompt": "Job title, e.g., Software Engineer, Product Manager, etc.",
        "dtype": "string",
    },
    "seniority_level": {
        "prompt": "Seniority level (e.g., Junior, Mid, Senior, Lead, Director, VP). Sometimes missing.",
        "dtype": "string",
    },
    "team_function": {
        "prompt": "Team or function (Engineering, Product, Design, GTM, Ops)",
        "dtype": "string",
    },
    "manager_org_priorities": {
        "prompt": "Manager or organization priorities (short text, sometimes missing or vague)",
        "dtype": "string",
    },
}

tables = {
    "employees": {
        "prompt": prompt,
        "columns": columns,
    }
}


def generate_employee_data():
    """
    Generate synthetic employee data for a tech company.
    Returns:
        DataFrame: A DataFrame containing the synthetic employee data.
    """
    df = mock.sample(
        tables=tables, sample_size=NUM_EMPLOYEES, model=f"{PROVIDER}/{MODEL}"
    )
    # Add employee_id (UUID)
    df["employee_id"] = [uuid.uuid4().int for _ in range(len(df))]

    # Reorder columns for clarity
    df = df[
        [
            "employee_id",
            "name",
            "job_title",
            "seniority_level",
            "team_function",
            "manager_org_priorities",
        ]
    ]
    return df


def write_to_csv(df, output_dir, file_name):
    """
    Write the DataFrame to a CSV file.

    Args:
        df (DataFrame): The DataFrame to write.
        output_dir (str): The directory where the CSV will be saved.
        file_name (str): The name of the file (without extension).
    """
    file_path = os.path.join(output_dir, file_name + ".csv")
    df.to_csv(file_path)
    print(f"Data saved to {file_path}")


def write_to_json(df, output_dir, file_name):
    """
    Write the DataFrame to a JSON file.

    Args:
        df (DataFrame): The DataFrame to write.
        output_dir (str): The directory where the JSON will be saved.
        file_name (str): The name of the file (without extension).
    """
    file_path = os.path.join(output_dir, file_name + ".json")
    df.to_json(file_path, orient="records", lines=True)
    print(f"Data saved to {file_path}")


def main():
    """
    Main function to generate synthetic employee data and save it to a CSV file.
    """
    df = generate_employee_data()
    write_to_csv(df, OUTPUT_DIR, FILE_NAME)
    print(
        f"\nGenerated {len(df)} employee records and saved to {OUTPUT_DIR}/{FILE_NAME}.csv"
    )


if __name__ == "__main__":
    main()
