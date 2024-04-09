contains data for finetuning / few-shot prompting and GPT-based model for the task of judging if there's hallucination in the given QA pair.

Due to the limited time and resources, we present the question to a weaker model, and let a stronger model(and human)judge if the answer is hallucinated or not.
Then we select some of the QA pairs to be used as the prompt for the GPT-4 model. It's strong enough to judge if the answer is hallucinated or not with the given prompt.