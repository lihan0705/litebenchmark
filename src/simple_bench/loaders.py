from datasets import load_dataset
import pandas as pd
from typing import List, Dict, Any, Optional

def _standardize(
    dataset_name: str,
    question: str,
    ground_truth: str,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    return {
        "dataset": dataset_name,
        "question": question,
        "ground_truth": ground_truth,
        "metadata": metadata or {}
    }

def load_gsm8k(split: str = "test", limit: Optional[int] = None, data_dir: Optional[str] = "./dataset") -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset.
    """
    ds = load_dataset("gsm8k", "main", split=split, cache_dir=data_dir)
    if limit:
        ds = ds.select(range(limit))
    
    data = []
    for item in ds:
        data.append(_standardize(
            dataset_name="gsm8k",
            question=item["question"],
            ground_truth=item["answer"],
            metadata={}
        ))
    return data

def load_hotpotqa(split: str = "validation", mode: str = "distractor", limit: Optional[int] = None, data_dir: Optional[str] = "./dataset") -> List[Dict[str, Any]]:
    """
    Load HotpotQA dataset.
    mode: 'distractor' or 'fullwiki'
    """
    ds = load_dataset("hotpot_qa", mode, split=split, trust_remote_code=True, cache_dir=data_dir)
    if limit:
        ds = ds.select(range(limit))
        
    data = []
    for item in ds:
        data.append(_standardize(
            dataset_name=f"hotpotqa_{mode}",
            question=item["question"],
            ground_truth=item["answer"],
            metadata={
                "type": item["type"],
                "level": item["level"],
                "supporting_facts": item["supporting_facts"],
                "context": item["context"]
            }
        ))
    return data

def load_gaia(split: str = "validation", limit: Optional[int] = None, data_dir: Optional[str] = "./dataset") -> List[Dict[str, Any]]:
    """
    Load GAIA dataset.
    """
    # GAIA typically has '2023_all', '2023_level1', etc. Using '2023_all' for now.
    ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split=split, cache_dir=data_dir)
    if limit:
        ds = ds.select(range(limit))
        
    data = []
    for item in ds:
        data.append(_standardize(
            dataset_name="gaia",
            question=item["Question"],
            ground_truth=item["Final_answer"],
            metadata={
                "level": item["Level"],
                "file_name": item.get("file_name", ""),
                "annotator_metadata": item.get("Annotator_Metadata", {})
            }
        ))
    return data

def load_mmmu(split: str = "validation", limit: Optional[int] = None, data_dir: Optional[str] = "./dataset") -> List[Dict[str, Any]]:
    """
    Load MMMU dataset.
    Note: MMMU has many configs (subjects). Loading a subset or specific subject might be needed.
    For simplicity, we might need to iterate over subjects or let user specify.
    Here we load a specific subject or 'all' if available, but MMMU usually requires subject.
    Let's default to a common one or handle 'all' if huggingface supports it.
    Actually MMMU on HF requires config name (subject).
    We will implement a helper to load multiple subjects or a default list.
    For this MVP, let's load 'Accounting' as a placeholder or try to load all if possible.
    Better: Load a few diverse subjects.
    """
    # MMMU structure is complex. Let's pick a few representative subjects for the default loader
    # or allow passing config.
    # For MVP, let's just load 'Accounting' to demonstrate.
    # Ideally we should iterate all subjects.
    
    subjects = ["Accounting", "Art", "Biology", "Business_Ethics", "Chemistry", "Computer_Science"]
    data = []
    
    for subject in subjects:
        try:
            ds = load_dataset("MMMU/MMMU", subject, split=split, cache_dir=data_dir)
            if limit:
                ds = ds.select(range(min(limit, len(ds))))
            
            for item in ds:
                data.append(_standardize(
                    dataset_name=f"mmmu_{subject}",
                    question=f"{item['question']} Options: {item['options']}",
                    ground_truth=item["answer"],
                    metadata={
                        "subject": subject,
                        "image": item.get("image", None), # Note: Image might be PIL object
                        "options": item["options"]
                    }
                ))
            if limit and len(data) >= limit:
                break
        except Exception as e:
            print(f"Failed to load MMMU subject {subject}: {e}")
            
    return data
