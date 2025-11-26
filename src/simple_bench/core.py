import pandas as pd
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Callable, Union
from tqdm.asyncio import tqdm
from .scorers import UniversalScorer
from ragas.dataset_schema import SingleTurnSample

class BenchmarkRunner:
    def __init__(self, dataset: List[Dict[str, Any]], agent_func: Callable[[str], Any]):
        """
        dataset: List of dicts with 'question', 'ground_truth', 'metadata'.
        agent_func: Async or sync function that takes a question string and returns a dict 
                    {'answer': str, ...} or just a string.
        """
        self.dataset = dataset
        self.agent_func = agent_func
        self.results = []
        self.scorer = UniversalScorer()

    async def _run_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        question = item["question"]
        ground_truth = item["ground_truth"]
        metadata = item.get("metadata", {})
        dataset_name = item.get("dataset", "")
        
        # Add dataset name to metadata for scorer
        metadata["dataset"] = dataset_name

        try:
            if asyncio.iscoroutinefunction(self.agent_func):
                response = await self.agent_func(question)
            else:
                response = self.agent_func(question)
        except Exception as e:
            response = {"answer": f"Error: {str(e)}"}

        # Normalize response
        if isinstance(response, str):
            answer = response
            rationale = ""
        elif isinstance(response, dict):
            answer = response.get("answer", "")
            rationale = response.get("rationale", "")
        else:
            answer = str(response)
            rationale = ""

        # Create Ragas sample for scoring
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            reference=ground_truth,
            metadata=metadata
        )

        # Score
        score = self.scorer.score(sample)

        return {
            "question": question,
            "ground_truth": ground_truth,
            "prediction": answer,
            "rationale": rationale,
            "score": score,
            "dataset": dataset_name,
            "metadata": metadata
        }

    async def run(self, concurrency: int = 10):
        """
        Run the benchmark with specified concurrency.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def sem_task(item):
            async with semaphore:
                return await self._run_single(item)

        tasks = [sem_task(item) for item in self.dataset]
        self.results = await tqdm.gather(*tasks, desc="Running Benchmark")
        return self

    def to_pandas(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results)

    def save(self, output_dir: str = "result", filename: str = None):
        """
        Save results to CSV and JSON.
        """
        if not self.results:
            print("No results to save.")
            return

        os.makedirs(output_dir, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}"
        
        # Remove extension if provided
        filename = os.path.splitext(filename)[0]
        
        # Save CSV
        df = self.to_pandas()
        csv_path = os.path.join(output_dir, f"{filename}.csv")
        df.to_csv(csv_path, index=False)
        
        # Save JSON (Summary + Details)
        json_path = os.path.join(output_dir, f"{filename}.json")
        report = self.report()
        report["details"] = self.results
        
        # Convert non-serializable objects in metadata if any
        def default_serializer(obj):
            return str(obj)

        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=default_serializer)
            
        print(f"Results saved to:\n- {csv_path}\n- {json_path}")

    def report(self) -> Dict[str, Any]:
        df = self.to_pandas()
        if df.empty:
            return {"error": "No results"}
        
        summary = {
            "total_samples": len(df),
            "average_score": df["score"].mean(),
            "accuracy": (df["score"] == 1.0).mean(),
        }
        
        # Group by dataset if multiple
        if "dataset" in df.columns and df["dataset"].nunique() > 1:
            summary["by_dataset"] = df.groupby("dataset")["score"].mean().to_dict()
            
        return summary
