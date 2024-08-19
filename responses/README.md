## Response comparisons between the original and finetuned models

For comparisons after finetuning, the training questions were presented to both the original 
Llama3_1-8B-Instruct model and the finetuned Llama3_1-8B-Instruct-AMD-python model.  

These files conain the questions and responses from both models on questions that were used in the training data:

 Technology: train_1111_responses.csv  
Python code: train_547_python_code_responses.csv  

These files contain the questions and responses from test questions that were not used for training (finetuning):

 Technology: test_150_responses.csv  
Python code: test_60_python_code_responses.csv  

Two methods are provided to make it easier to compare model output from before and after finetuning. 
A single question row can be compared with the following python script.
python ../python/question_row_compare.py <csv file> --row <row number>

To compare consecutive rows starting at a question row number, the following script can be used.
A pause for a space bar will occur after each question row comparison.
scan_responses.sh <csv file> <start row number>
