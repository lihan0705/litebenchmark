import asyncio
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from simple_bench import BenchmarkRunner, load_gsm8k, UniversalScorer

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

async def dummy_agent(question: str):
    messages = [HumanMessage(content=question)]
    response = await llm.ainvoke(messages)
    return {"answer": response.content}

async def main():
    print("Loading GSM8K with custom data_dir='./dataset'...")
    try:
        # Test custom data_dir
        dataset = load_gsm8k(split="test", limit=5, data_dir="./dataset")
        print(f"Loaded {len(dataset)} items.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Running Benchmark...")
    runner = BenchmarkRunner(dataset, dummy_agent)
    await runner.run()
    
    print("Generating Report...")
    report = runner.report()
    print("Report Summary:", report)
    
    # Test save method
    print("Saving results to './result'...")
    runner.save(output_dir="./result", filename="verification_run")
    
    df = runner.to_pandas()
    print("DataFrame Head:")
    print(df.head())
    
    # Check if scoring works (even if score is 0)
    if "score" in df.columns:
        print("Scoring column exists.")
    else:
        print("Scoring column MISSING!")

if __name__ == "__main__":
    asyncio.run(main())
