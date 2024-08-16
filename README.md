# LLM agent for AMD Technologies and Python

This project built an expert LLM agent trained with current knowledge of AMD Technologies (with emphasis on GPUs) and Python for use during on-premises development. It finetuned the Llama 3_1-8B-Instruct LLM.

A corpus was constructed with more recent documentation for use in a retrieval-augmented generation (RAG) implementation. Queries can extract relevant information from a vector store (FAISS in this case) to provide additional context for the larger teacher model on questions. The corpus for RAG consisted of publicly available AMD and related open-source documentation including PDFs, websites, blogs, press releases (for 2023 and 2024) and github repositories. 

To create a training set for finetuning, questions on related topics and python coding were requested from both Google Gemini and ChatGPT-4o. In all, a total of 1111 technology questions and 547 coding questions were generated. With the use of RAG, these questions were answered via API using Google Gemini 1.5 Pro. The questions and answers saved as CSV files were converted into two Alpaca training sets with instruction-output pairs.

Llama 3_1-8B-Instruct was selected as the base model, due to a relatively small size and a recent release. Torchtune was used for full finetuning, on a single Instinct MI210 GPU. The 1658 questions and answers were presented for 5 epochs in total

Additional questions were generated using an OpenAI GPT, which was given the previously generated questions as context and asked to generate related new and unique questions for the testing. This set was composed of 150 technology questions and 60 python coding questions. The training questions and the test questions were presented to both the original Llama3_1-8G-Instruct model and the finetuned Llama3_1-8G-Instruct-AMD-python model for comparison.

The original Llama 3.1 model does quite well on responses and python code examples. However, the finetuned model seems to produce more specific, although briefer and more accurate responses. Particularly, the original model will, for example, refer to older Instinct GPU hardware or older ROCm releases in responses. The finetuned model will more likely (although not always) respond with more current versions of hardware and software. Sometimes the produced python code also seems to be more current, but detailed analysis is required to check for both the accuracy and the successful execution of the code on AMD and other hardware.

Complete side-by-side comparisons in spreadsheets are available in the responses directory.

The finetuned LLM is on HuggingFace at: davidsi/Llama3_1-8B-Instruct-AMD-python
