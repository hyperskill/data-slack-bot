import re
import json
import itertools

from tenacity import retry, stop_after_attempt, wait_exponential

from hyperskillai_api import HyperskillAIAPI

class PromptsGenerator:
    # K is a constant factor that determines how much rating change
    K = 32

    # Number of times to retry a call to the ranking model if it fails
    N_RETRIES = 3

    # This determines how many test cases to generate.
    # The higher, the more expensive and long, but the better the results will be
    __NUMBER_OF_TEST_CASES = 3

    # This determines how many candidate prompts to generate.
    # The higher, the more expensive and long, but the better the results will be
    __NUMBER_OF_PROMPTS = 3

    __CANDIDATE_MODEL_TEMPERATURE = 0.9
    __GENERATION_MODEL_TEMPERATURE = 0.8
    __TEST_CASE_MODEL_TEMPERATURE = 0.8
    __RANKING_MODEL_TEMPERATURE = 0.5

    __GENERATION_MODEL_MAX_TOKENS = 800


    def __init__(self, ai_api: HyperskillAIAPI):
        self.__ai_api = ai_api


    def generate_optimal_prompt(self, description: str, input_variables: list[dict[str, str]]) -> str:
        test_cases = self.__generate_test_cases(description, input_variables, self.__NUMBER_OF_TEST_CASES)
        prompts = self.__generate_candidate_prompts(description, input_variables, test_cases, self.__NUMBER_OF_PROMPTS)
        prompt_ratings = self.__test_candidate_prompts(test_cases, description, prompts)

        # Returning the prompt with the best ELO rating
        return sorted(prompt_ratings.items(), key=lambda item: item[1], reverse=True)[0][0]


    def __generate_test_cases(
        self,
        description: str,
        input_variables: list[dict[str, str]],
        num_test_cases: int = 5
    ) -> list[dict]:
        variable_descriptions = "\n".join(f"{var['variable']}: {var['description']}" for var in input_variables)

        messages = [
            {
                "role": "system",
                "content": f"""You are an expert at generating test cases for evaluating AI-generated content.
                Your task is to generate a list of {num_test_cases} test case prompts based on the given description and input variables.
                Each test case MUST be a JSON object with a 'test_design' field containing the overall idea of this test case, and a list of additional JSONs for each input variable, called 'variables'.
                The test cases must be diverse, covering a range of topics and styles relevant to the description.
                Here are the input variables and their descriptions:
                {variable_descriptions}
                You MUST return the test cases as a JSON list, with no other text or explanation.""",
            },
            {
                "role": "user",
                "content": f"Description: {description.strip()}\n\nGenerate the test cases. Make sure they are really, really great and diverse:",
            }
        ]

        response_text = self.__ai_api.get_chat_completion(
            messages=messages,
            max_tokens=1500,
            temperature=self.__CANDIDATE_MODEL_TEMPERATURE
        )

        test_cases = json.loads(response_text)
        return test_cases


    def __generate_candidate_prompts(
        self,
        description: str,
        input_variables: list[dict[str, str]],
        test_cases: list[dict],
        number_of_prompts: int
    ) -> list[str]:
        variable_descriptions = "\n".join(f"{var['variable']}: {var['description']}" for var in input_variables)

        messages = [
            {
                "role": "system",
                "content": f"""Your job is to generate system prompts for LLM, given a description of the use-case, some test cases/input variable examples that will help you understand what the prompt will need to be good at.
    The prompts you will be generating will be for freeform tasks, such as generating a landing page headline, an intro paragraph, solving a math problem, etc.
    In your generated prompt, you MUST describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output.
    <most_important>Make sure to incorporate the provided input variable placeholders into the prompt, using placeholders like {{{{VARIABLE_NAME}}}} for each variable. Ensure you place placeholders inside four squiggly lines like {{{{VARIABLE_NAME}}}}. At inference time/test time, we will slot the variables into the prompt, like a template.</most_important>
    Be creative with prompts to get the best possible results! The AI knows it's an AI -- you DON'T NEED to tell it this.
    You will be graded based on the performance of your prompt... but DON'T CHEAT! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified!
    Here are the input variables and their descriptions:
    {variable_descriptions}
    Most importantly, output NOTHING but the prompt (with the variables contained in it like {{{{VARIABLE_NAME}}}}). DO NOT include anything else in your message.""",
            },
            {
                "role": "user",
                "content": f"Here are some test cases:`{test_cases}`\n\nHere is the description of the use-case: `{description.strip()}`\n\nRespond with your flexible system prompt, and nothing else. Be creative, and remember, the goal is not to complete the task, but WRITE A PROMPT THAT WILL COMPLETE THE TASK!",
            }
        ]

        prompts = []

        for _ in range(number_of_prompts):
            response_text = self.__ai_api.get_chat_completion(
                messages=messages,
                max_tokens=1500,
                temperature=self.__CANDIDATE_MODEL_TEMPERATURE
            )

            prompts.append(self.__remove_first_line(response_text))

        return prompts


    def __test_candidate_prompts(
        self,
        test_cases: list[dict],
        description: str,
        prompts: list[str]
    ) -> dict[str, int]:
        # Initialize each prompt with an ELO rating of 1200
        prompt_ratings = {prompt: 1200 for prompt in prompts}

        for prompt_1, prompt_2 in itertools.combinations(prompts, 2):
            for test_case in test_cases:
                generation_1 = self.__get_generation(prompt_1, test_case)
                generation_2 = self.__get_generation(prompt_2, test_case)

                # Rank the outputs
                score_1 = self.__get_score(description, test_case, generation_1, generation_2)
                score_2 = self.__get_score(description, test_case, generation_2, generation_1)

                # Convert scores to numeric values
                score_1 = 1 if score_1 == 'A' else 0 if score_1 == 'B' else 0.5
                score_2 = 1 if score_2 == 'B' else 0 if score_2 == 'A' else 0.5

                # Average the scores
                score = (score_1 + score_2) / 2

                # Update ELO ratings
                rating_1, rating_2 = prompt_ratings[prompt_1], prompt_ratings[prompt_2]
                rating_1, rating_2 = self.__update_elo(rating_1, rating_2, score)
                prompt_ratings[prompt_1], prompt_ratings[prompt_2] = rating_1, rating_2

        return prompt_ratings


    @retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=70))
    def __get_generation(self, prompt: str, test_case: dict) -> str:
        # Replace variable placeholders in the prompt with their actual values from the test case
        for var_dict in test_case['variables']:
            for variable_name, variable_value in var_dict.items():
                prompt = prompt.replace(f"{{{{{variable_name}}}}}", variable_value)

        messages = [
            {
                "role": "system",
                "content": "Complete the task perfectly!",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]

        generation_text = self.__ai_api.get_chat_completion(
            messages=messages,
            temperature=self.__GENERATION_MODEL_TEMPERATURE,
            max_tokens=self.__GENERATION_MODEL_MAX_TOKENS,
        )

        return generation_text


    # Get Score - retry up to N_RETRIES times, waiting exponentially between retries.
    @retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=70))
    def __get_score(self, description: str, test_case: dict, pos_1: str, pos_2: str) -> float:
        messages = [
            {
                "role": "system",
                "content": f"""Your job is to rank the quality of two outputs generated by different prompts. The prompts are used to generate a response for a given task.
    You will be provided with the task description, input variable values, and two generations - one for each system prompt.
    Rank the generations in order of quality. If Generation A is better, respond with 'A'. If Generation B is better, respond with 'B'.
    Remember, to be considered 'better', a generation must not just be good, it must be noticeably superior to the other.
    Also, keep in mind that you are a very harsh critic. Only rank a generation as better if it truly impresses you more than the other.
    Respond with your ranking ('A' or 'B'), and nothing else. Be fair and unbiased in your judgement.""",
            },
            {
                "role": "user",
                "content":  f"""
                    Task: {description.strip()}
                    Variables: {test_case['variables']}
                    Generation A: {self.remove_first_line(pos_1)}
                    Generation B: {self.remove_first_line(pos_2)}""",
            }
        ]

        score = self.__ai_api.get_chat_completion(
            messages=messages,
            max_tokens=1,
            temperature=self.__RANKING_MODEL_TEMPERATURE
        )

        return score


    def __update_elo(self, rating_1: float, rating_2: float, score: float) -> float:
        expected_1 = self.__expected_score(rating_1, rating_2)
        expected_2 = self.__expected_score(rating_2, rating_1)
        return rating_1 + self.K * (score - expected_1), rating_2 + self.K * ((1 - score) - expected_2)


    @staticmethod
    def __remove_first_line(test_string: str) -> str:
        if test_string.startswith("Here") and test_string.split("\n")[0].strip().endswith(":"):
            return re.sub(r'^.*\n', '', test_string, count=1)
        return test_string


    @staticmethod
    def __expected_score(rating_l: float, rating_r: float) -> float:
        return 1 / (1 + 10 ** ((rating_r - rating_l) / 400))
