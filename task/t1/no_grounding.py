import asyncio
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `flow_diagram.png` to see the flow of app

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY: 
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }


# 1. Create AzureChatOpenAI client
client = AzureChatOpenAI(
    api_key=SecretStr(API_KEY),
    api_version="",
    azure_endpoint=DIAL_URL,
    model="gpt-4o",
    temperature=0.0,
)
#    hint: api_version set as empty string if you gen an error that indicated that api_version cannot be None
# 2. Create TokenTracker
token_tracker = TokenTracker()

def join_context(context: list[dict[str, Any]]) -> str:
    # You cannot pass raw JSON with user data to LLM (" sign), collect it in just simple string or markdown.
    # You need to collect it in such way:
    # User:
    #   name: John
    #   surname: Doe
    """
    Converts a list of user dicts into a markdown-like human-readable string.
    Each user starts with `User:` followed by indented key: value pairs.
    """
    if not context:
        return "No user data available."
    lines = []
    for user in context:
        lines.append("User:")
        for key, value in user.items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)



async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    # 2. Generate response (use `ainvoke`, don't forget to `await` the response)
    response = await client.ainvoke(messages)
    # 3. Get usage (hint, usage can be found in response metadata (its dict) and has name 'token_usage', that is also
    #    dict and there you need to get 'total_tokens')
    token_usage = response.response_metadata.get('token_usage', {}).get('total_tokens', 0)
    # 4. Add tokens to `token_tracker`
    token_tracker.add_tokens(token_usage)
    # 5. Print response content and `total_tokens`
    print(f"Response content: {response.content}")
    print(f"Total tokens: {token_usage}")
    # 5. return response content
    return response.content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")

        #TODO:
        # 1. Get all users (use UserClient)
        users = UserClient().get_all_users()
        # 2. Split all users on batches (100 users in 1 batch). We need it since LLMs have its limited context window
        batch_size = 100
        user_batches = [users[i:i + batch_size] for i in range(0, len(users), batch_size)]
        # 3. Prepare tasks for async run of response generation for users batches:
        #       - create array tasks
        #       - iterate through `user_batches` and call `generate_response` with these params:
        #           - BATCH_SYSTEM_PROMPT (system prompt)
        #           - User prompt, you need to format USER_PROMPT with context from user batch and user question
        tasks = []
        for user_batch in user_batches:
            # Prepare context string for this batch
            batch_context_str = join_context(user_batch)
            # Format the user prompt using USER_PROMPT template with batch context and user question
            user_prompt = USER_PROMPT.format(context=batch_context_str, query=user_question)
            # Create coroutine for generating response for this batch
            task = generate_response(
                system_prompt=BATCH_SYSTEM_PROMPT, 
                user_message=user_prompt)
            tasks.append(task)
        # 4. Run task asynchronously, use method `gather` form `asyncio`
        batch_result = await asyncio.gather(*tasks)
        # 5. Filter results on 'NO_MATCHES_FOUND' (see instructions for BATCH_SYSTEM_PROMPT)
        filtered_results = [res for res in batch_result if res.strip() != 'NO_MATCHES_FOUND']
        # 5. If results after filtration are present:
        if filtered_results:
        #       - combine filtered results with "\n\n" spliterator
            combined_results = "\n\n".join(filtered_results)
        #       - generate response with such params:
        #           - FINAL_SYSTEM_PROMPT (system prompt)
        #           - User prompt: you need to make augmentation of retrieved result and user question
            await generate_response(
                system_prompt=FINAL_SYSTEM_PROMPT,
                user_message=f"## SEARCH RESULTS:\n{combined_results}\n\n## ORIGINAL QUERY:\n{user_question}"
            )
        # 6. Otherwise print the info that `No users found matching`
        else:
            print("No users found matching the search criteria.")
        # 7. In the end print info about usage, you will be impressed of how many tokens you have used. (imagine if we have 10k or 100k users 😅)
        summary = token_tracker.get_summary()
        print("\n--- Token Usage Summary ---")
        print(f"Total tokens used: {summary['total_tokens']}")
        print(f"Number of batches processed: {summary['batch_count']}")
        print(f"Tokens used per batch: {summary['batch_tokens']}")




if __name__ == "__main__":
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation