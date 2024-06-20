# Unveiling and Mitigating Bias in Mental Health Analysis with Large Language Models

## Objectives

This work aims to systematically evaluate biases across seven social factors using ten LLMs with different prompting methods on eight diverse datasets. We also examine bias mitigation in LLMs by proposing a set of fairness-aware prompts. Our results show that GPT-4 achieves the best overall balance in performance and fairness among LLMs, although it still lags behind domain-specific models like MentalRoBERTa in some cases. Additionally, tailored fair prompts can effectively mitigate bias in mental health applications, highlighting the great potential for fair analysis in this field.

## Models

We divide the models used in our experiments into two major categories. 
- Discriminative BERT-based models: BERT/RoBERTa (Kenton and Toutanova, 2019; Liu et al., 2019) and MentalBERT/MentalRoBERTa (Ji et al., 2022b).
  
- LLMs of varying sizes, including TinyLlama-1.1B-Chat-v1.0 (Zhang et al., 2024), Phi-3-mini-128k-instruct (Abdin et al., 2024), gemma-2b-it, gemma-7b-it (Team et al., 2024), Llama-2-7b-chat-hf, Llama-2-13bchat-hf (Touvron et al., 2023), MentaLLaMA-chat7B, MentaLLaMA-chat-13B (Yang et al., 2024),
Llama-3-8B-Instruct (AI@Meta, 2024), and GPT4 (Achiam et al., 2023).

GPT-4 is accessed through the OpenAI API, while the remaining models are loaded from Hugging Face.
