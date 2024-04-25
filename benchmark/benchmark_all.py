#import debugpy;debugpy.connect(5678)
from argparse import Namespace
from MMLU_Medical.evaluate import main as mmlu_medical_main
SIMPLE_MCQA_TASKS=[
    'MedQA',
    'MedMCQA',
    'MMLU',
    'MLEC-QA'
    
]
GENERATION_TASKS=[
    'PubMedQA',
    'Medication_QA_MedInfo2019',
    'LiveQA_MedicalTask_TREC2017',
    'iCliniq-10k',
    'HealthSearchQA'
]
def benchmark():
    print("Benchmarking start")


def benchmark_MMLU():
    
    print("Benchmarking MMLU start:")
    
if __name__ == "__main__":
    mmlu_args = Namespace()
    mmlu_args.ntrain = 0
    mmlu_args.data_dir = "benchmark/MMLU_Medical/data"
    mmlu_args.save_dir = "benchmark/results/MMLU_Medical"
    mmlu_args.engine = ["davinci"]
    
    mmlu_medical_main(mmlu_args)