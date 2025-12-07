"""
DeepAgent Architecture Implementation - Contains Root Agent and Math SubAgent
Using the correct deepagents SubAgentMiddleware design pattern
All functionality is implemented in system prompts, no additional tools needed
"""

from typing import List, Dict, Any
from langchain.agents import create_agent
from deepagents.middleware.subagents import SubAgentMiddleware


# Create Math SubAgent Configuration
# All math functionality is implemented in the system prompt, no additional tools needed
math_subagent_config = {
    "name": "math",
    "description": "Math problem solving sub-agent, skilled in arithmetic, algebra, statistics, and unit conversion",
    "system_prompt": """
You are a professional math solver assistant. Your task is:
1. Accurately understand math problems
2. Provide clear problem-solving processes and answers
3. For complex problems, recommend using more specialized tools

Math capabilities include:
1. Basic arithmetic operations (addition, subtraction, multiplication, division, exponentiation, etc.)
2. Algebraic equation solving
3. Statistical calculations (mean, median, standard deviation, etc.)
4. Unit conversions (length, weight, temperature, etc.)

Problem-solving steps:
1. Carefully analyze the problem type
2. Calculate step by step
3. Verify the reasonableness of the results
4. Clearly display the problem-solving process

Examples:
Question: Calculate 15 + 23 * 2
Answer: According to operation priority, multiply first then add. 23 * 2 = 46, then 15 + 46 = 61. The answer is 61.

Question: Find the average of the dataset [1, 3, 5, 7, 9]
Answer: Average = (1+3+5+7+9)/5 = 25/5 = 5. The answer is 5.

Question: Convert 100 centimeters to meters
Answer: 1 meter = 100 centimeters, so 100 centimeters = 1 meter. The answer is 1 meter.
""",
    "tools": [],  # No additional tools needed, all functionality is implemented in the system prompt
    "model": "gemini-2.5-flash",
    "middleware": [],
}


# Create Root Agent
def create_root_agent():
    """
    Create root agent with math sub-agent middleware
    """
    
    # Create sub-agent middleware
    subagent_middleware = SubAgentMiddleware(
        default_model="gemini-2.5-flash",
        default_tools=[],
        subagents=[math_subagent_config]
    )
    
    # Create root agent
    agent = create_agent(
        model="gemini-2.5-flash",
        system_prompt="""
You are an intelligent task coordinator. Your main responsibilities are:
1. Analyze user requests and determine if sub-agents need to be called
2. Delegate math-related tasks to the math sub-agent
3. Integrate sub-agent results and provide final answers

When encountering math problems, please call the math sub-agent.
Math problems include but are not limited to:
- Basic arithmetic operations
- Algebraic equation solving
- Statistical calculations
- Unit conversions
""",
        middleware=[
            subagent_middleware
        ],
    )
    
    return agent


# Set Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBLcu0U5vqa3uPVPBlEMD9ojoIoNuVVZk4"
os.environ["OPENAI_API_KEY"] = "test" # Just in case

# Usage example
if __name__ == "__main__":
    # Create root agent
    try:
        agent = create_root_agent()
        print("=== DeepAgent System Started ===")
        print("Root agent created with math sub-agent")
        
        # Test invocation
        query = "Calculate 15 + 23 * 2"
        print(f"\nTesting Query: {query}")
        result = agent.invoke(query)
        print(f"Result: {result}")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Please ensure 'deepagents' and 'langchain' are installed.")
    except Exception as e:
        print(f"Runtime Error: {e}")