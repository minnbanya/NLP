# NLP A6
 AIT NLP Assignment 6

- [Student Information](#student-information)
- [Model evaluation and comparison](#model-evaluation-and-comparison)
- [Challenges and improvement](#challenges-and-improvements)



## Student Information
Name - Minn Banya  
ID - st124145
 

## Model evaluation and comparison
| Student Layer | Training Loss | Validation Loss | Validation Accuracy |
|---------------|---------------|-----------------|---------------------|
| Top-K Layer   |      0.2473         |        0.7968         |        68.30%             |
| Bottom-K Layer|        0.2452       |          0.7968       |          68.30%           |
| Odd Layer     |       0.2447        |         0.7968        |           68.30%          |
| Even Layer    |       0.2444        |          0.7968       |          68.30%           |

| Teacher model | Training Loss | Validation Loss | Validation Accuracy |
|---------------|---------------|-----------------|---------------------|
| BertForSequenceClassification   |      0.0314         |        1.3048         |        71.40%             |

It would appear that the selection of initial layers had no impact on the final performance of the models. Only the top K layer model showed learning with the subsequent models remaining at the same final loss and accuracy across the board. More experimentation would be necessary to see whether the layer selection is truely inconsequencetial. It does however seem to reach just below the teacher model's performance, even with having 40% less parameters.

## Challenges and improvements
The challenges faced were mainly how to select the right layers according to the instructions. The fact that there is no difference in the performance of the model shows the insignificance of the initial layer selection. The size of the training data also proved to be too much for the limited computational power at my disposal, forcing me to limit the number of rows to 10,000.

As such, a slight improvement to the performance of the model could come from performing exhastive search for all possible layer selection permutations to see which one performs the best. Increasing the size of the training data would also be another way of increasing the performance of the model.