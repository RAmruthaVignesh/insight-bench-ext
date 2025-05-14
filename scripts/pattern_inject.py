from openai import OpenAI
import json
import os
import shutil
import re
import subprocess
import pandas as pd


class PatternInjector:
    def __init__(self, api_key: str = None):
        """Initialize the PatternInjector with OpenAI API key.

        Args:
            api_key: OpenAI API key. If not provided, will try to get from OPENAI_API_KEY environment variable.
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "OpenAI API key not provided and OPENAI_API_KEY environment variable not set"
                )
        self.client = OpenAI(api_key=api_key)

    def get_inject_codes(self, patterns: str) -> dict:
        """Get the code to inject the pattern into the data.

        Args:
            patterns: The patterns to inject. It is a json file with the following format:
            {
                "kpis": [...],
                "patterns": [
                    {
                        "id": "pattern_id",
                        "pattern": "description of the pattern",
                        "columns_involved": ["col1", "col2", ...],
                        "reasoning": "explanation of why this pattern is useful",
                        "relevance_to_kpi": "how this pattern helps with the task",
                        "benchmark_value": "value to test against"
                    }
                ],
                "Answers": [
                    {
                        "question": "question text",
                        "answer_after_injection": "answer text",
                        "caused_by_pattern": "pattern_id"
                    }
                ]
            }

        Returns:
            The code to inject the pattern into the data.
        """

        print("Started getting inject codes ...")

        patterns_dict = json.loads(patterns)
        pattern_info = patterns_dict.get("patterns", [])[0]  # Get the single pattern
        pattern_id = pattern_info.get("pattern_index", "")

        # Find questions specific to this pattern from Answers
        questions = [
            q
            for q in patterns_dict.get("answers", [])
            if q.get("caused_by_pattern") == pattern_id
        ]

        # Format questions and answers
        qa_context = ""
        if questions:
            qa_context = "\nQuestions and Answers for this pattern:\n"
            for q_idx, qa in enumerate(questions, 1):
                qa_context += f"\nQ{q_idx}: {qa.get('question', '')}\n"
                qa_context += f"A{q_idx}: {qa.get('answer_after_injection', '')}\n"

        # Strip serial numbers from column names
        columns = [
            col.split(". ", 1)[-1] if ". " in col else col
            for col in pattern_info.get("columns_involved", [])
        ]
        pattern_description = pattern_info.get("pattern", "")
        reasoning = pattern_info.get("reasoning", "")
        relevance = pattern_info.get("relevance_to_kpi", "")

        function_name = "modify_" + "_".join(columns)
        columns_str = ""
        for column in columns:
            columns_str += f"'{column}\n"

        prompt = f"""
        You are a highly skilled data scientist. Your task is to reason through a pattern injection task and generate a valid, executable Python function to apply the specified transformation.

        You are given a pandas DataFrame named `df` with the following columns:
                    
            {columns_str}

        Your goal is to implement a data transformation that introduces the following pattern:

        - **Pattern**: {pattern_description}
        - **Reasoning**: {reasoning}
        - **Relevance**: {relevance}
        {qa_context}

        Please follow this **step-by-step reasoning process** internally before generating code:
        1. Identify which of the columns in `{columns_str}` are directly involved in the pattern.
        2. Determine what type of transformation needs to be applied (e.g., numeric shift, conditional flag, group-based anomaly).
        3. Ensure that the transformation **modifies the data while preserving the answer to the question** in the provided context.
        4. Choose appropriate Python libraries and operations to implement the transformation.
        5.  Validate that the logic is self-contained and does not rely on any undefined variables or external dependencies.
        6. Review the provided questions and answers to ensure your transformation aligns with and maintains the expected behavior.

        Then, generate a complete **Python function** with the following signature:

        ```python
        def {function_name}(df: pd.DataFrame) -> pd.DataFrame:
        ```
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a data pattern injection coding expert. Your task is to write correct and simple codes to inject the given patterns to help accomplish specific analytics tasks. Always respond with valid Python Code.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        output = {
            f"Pattern_{pattern_id}_{'_'.join(columns)}": response.choices[
                0
            ].message.content.strip()
        }
        print(
            f"Finished getting inject codes for pattern {pattern_id} on columns: {'_'.join(columns)}"
        )

        return output

    def test_inject(
        self, injected_data: pd.DataFrame, question: str, expected_answer: str
    ) -> bool:
        """Test if the injected pattern maintains the expected answer to the question.

        Args:
            injected_data: The data after pattern injection
            question: The question to test
            expected_answer: The expected answer to the question

        Returns:
            bool: True if the answer matches the expected answer, False otherwise
        """
        print(f"\nTesting question: {question}")
        print(f"Expected answer: {expected_answer}")

        # Create a prompt to generate code that answers the question
        prompt = f"""
        You are a data analysis expert. Your task is to write code that answers the following question about a pandas DataFrame df.

        Question: {question}
        Expected Answer: {expected_answer}
        df: {injected_data}

        The DataFrame is named 'df' and contains the data after pattern injection.
        Write a Python function that:
        1. Takes the DataFrame as input
        2. Analyzes the data to answer the question
        3. Returns the answer in a clear, concise format

        The code should be self-contained and only use the DataFrame 'df'.
        Return only the Python code that answers the question, no explanations.
        The function should be named 'answer_question' and take a pandas DataFrame as input.
        """

        try:
            # Get code to answer the question
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analysis coding expert. Your task is to write code that answers questions about data. Always respond with valid Python Code.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            answer_code = response.choices[0].message.content.strip()
            match = re.search(r"```python(.*?)```", answer_code, re.DOTALL)
            if match:
                answer_code = match.group(1).strip()

            print(f"\nGenerated code to answer the question:\n{answer_code}")

            # Create a namespace to execute the code
            namespace = {}

            # Execute the code to define the function
            exec(answer_code, namespace)

            # Get the function from the namespace
            answer_function = namespace.get("answer_question")

            if answer_function is None:
                print(
                    "Error: Could not find answer_question function in the generated code"
                )
                return False

            # Execute the function on the injected data
            actual_answer = answer_function(injected_data)
            print(f"Actual answer from code execution: {actual_answer}")

            # Create a prompt to check if the answers match
            check_prompt = f"""
            You are a data analysis expert. Your task is to determine if two answers to a data analysis question are equivalent.

            Question: {question}
            Expected Answer: {expected_answer}
            Actual Answer: {actual_answer}

            Please analyze if the actual answer matches the expected answer.
            Consider:
            1. Are the values numerically equivalent?
            2. Are the units and format consistent?
            3. Are the conclusions or findings the same?
            4. Are there any minor differences that don't affect the meaning?

            Respond with a clear explanation of whether the answers are equivalent, and why.
            """

            # Check if the answers match
            check_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analysis expert. Your task is to compare and validate answers to data analysis questions.",
                    },
                    {"role": "user", "content": check_prompt},
                ],
            )

            validation_result = check_response.choices[0].message.content.strip()
            print(f"\nValidation result: {validation_result}")

            # Extract the boolean result from the validation
            is_valid = (
                "equivalent" in validation_result.lower()
                or "match" in validation_result.lower()
            )
            return is_valid

        except Exception as e:
            print(f"Error testing injection: {str(e)}")
            return False

    def inject_patterns(
        self,
        base_df: pd.DataFrame,
        pattern_codes: dict,
        hash_id: str = None,
    ) -> pd.DataFrame:
        """Inject the patterns into the data.

        Args:
            base_df: The base DataFrame to inject the patterns into.
            pattern_codes: The pattern codes to inject. It is a dictionary with the following format:
            {
                "name1": "code to inject for pattern1",
                "name1": "code to inject for pattern2",
                ...
            }
            hash_id: The hash ID to use for the temp directory. If not provided, will use "default".
        """

        print("Started injecting patterns ...")

        # Step 1: Create temp directory inside results/{hash_id}/codefiles/
        if hash_id is None:
            hash_id = "default"
        temp_dir = os.path.join("results", hash_id, "codefiles")
        os.makedirs(temp_dir, exist_ok=True)

        # Step 2: Handle input data
        if isinstance(base_df, pd.DataFrame):
            df = base_df.copy()
            filename = "temp_data.csv"
            temp_csv_path = os.path.join(temp_dir, filename)
            df.to_csv(temp_csv_path, index=False)
        else:
            raise ValueError(
                "base_df should be a pandas DataFrame. Please provide a valid DataFrame."
            )

        # Step 3: Create Python scripts for each column
        for pattern_name, raw_code in pattern_codes.items():
            print(f"Injecting pattern: {pattern_name}")

            match = re.search(r"```python(.*?)```", raw_code, re.DOTALL)
            code = match.group(1).strip() if match else raw_code.strip()

            code = re.sub(
                r"^\s*import\s+pandas\s+as\s+pd\s*\n?", "", code, flags=re.MULTILINE
            )

            func_name = f"modify_" + "_".join(pattern_name.split("_")[1:]).replace(
                " ", "_"
            )

            # Ensure function name is consistent in the code
            code = re.sub(
                r"def\s+[a-zA-Z0-9_]+\(df: pd\.DataFrame\)",
                f"def {func_name}(df: pd.DataFrame)",
                code,
            )

            final_code = "import pandas as pd\n" + code.strip() + "\n"
            final_code += """if __name__ == "__main__":\n"""
            final_code += f"""   df = pd.read_csv("temp_data.csv")\n"""
            final_code += f"""   df = {func_name}(df)\n"""
            final_code += f"""   df.to_csv("temp_data.csv", index=False)"""

            # Create script in codefiles directory
            script_name = f"{func_name}.py"
            script_path = os.path.join(temp_dir, script_name)

            # Retry logic for pattern injection
            max_retries = 3
            retry_count = 0
            last_error = None

            while retry_count < max_retries:
                try:
                    # Always write the script file (in case of previous failures)
                    with open(script_path, "w") as f:
                        f.write(final_code)
                    print(f"Created/Updated script for pattern: {pattern_name}")

                    # Step 4: Run the script
                    subprocess.run(["python3", script_name], check=True, cwd=temp_dir)

                    # Update the DataFrame with the modified data
                    df = pd.read_csv(temp_csv_path)
                    print(f"Successfully injected pattern: {pattern_name}")
                    break

                except Exception as e:
                    last_error = e
                    retry_count += 1
                    print(
                        f"Error injecting pattern (attempt {retry_count}/{max_retries}): {str(e)}"
                    )

                    if retry_count < max_retries:
                        # Regenerate code with error information
                        error_prompt = f"""
                        Previous code failed with error: {str(e)}
                        Please fix the following code to handle the error:
                        {final_code}
                        """

                        response = self.client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a data pattern injection coding expert. Your task is to fix the code that failed with the given error. Always respond with valid Python Code.",
                                },
                                {"role": "user", "content": error_prompt},
                            ],
                        )

                        # Update the code with the fixed version
                        fixed_code = response.choices[0].message.content.strip()
                        match = re.search(r"```python(.*?)```", fixed_code, re.DOTALL)
                        if match:
                            fixed_code = match.group(1).strip()

                        # Update final_code with the fixed version
                        final_code = "import pandas as pd\n" + fixed_code.strip() + "\n"
                        final_code += """if __name__ == "__main__":\n"""
                        final_code += f"""   df = pd.read_csv("temp_data.csv")\n"""
                        final_code += f"""   df = {func_name}(df)\n"""
                        final_code += f"""   df.to_csv("temp_data.csv", index=False)"""

                        print(f"Regenerated code for pattern: {pattern_name}")
                    else:
                        print(
                            f"Failed to inject pattern after {max_retries} attempts. Last error: {str(last_error)}"
                        )
                        print("Continuing with next pattern...")
                        continue

        print("Finished injecting patterns for all patterns.")

        # Step 5: Clean up
        # Don't remove the directory since we want to keep the code files
        # shutil.rmtree(temp_dir)

        return df
