# ElectionResultsPredictor
Using the provided dataset, implemented 4 different classification algorithms (K-Nearest Neighbor, Perceptron ,Multi Layer perceptron, Decision Tree) to determine if a candidate wins an election. The focus of this dataset is on how financial resources affected the result of the campaign.

Used the Pandas and Numpy library.

<b>Data:</b>
The attached csv file contains 1500 candidates and information about their campaign. The focus of this dataset is how financial resources affected the result of the campaign. A full description of the dataset is given at <a> https://classic.fec.gov/finance/disclosure/metadata/metadataforcandidatesummary.shtml </a>. I used 5 features to predict if the candidate won or lost, namely : Net Operating Expenditure , Net Contributions , Total Loan , Candidate Office ,
Candidate Incumbent Challenger Open Seat

For a given candidate, used the winner column (Y/N) as the target.

<b>Implementation:</b>
Each model was tested in three steps. These functions are in the provided python file.
1. model.train(train_features, train_labels) for maximum of 1 minute
2. predictions = model.predict(test_features)
3. evaluate(predictions, test_labels)

<b>K-Nearest Neighbor:</b>
Used Euclidean distance between the features. For categorical data, converted each category to
its own column with a value of 1 if the data is that category or otherwise a 0 (Called One Hot
Encoding). Chose a k value and used majority voting to determine the class.

<b>Perceptron and Multi-Layer perceptron (MLP):</b>
As with KNN, used a one hot encoding of the input data. For the perceptron, multiplied (dot
multiplied) the inputs by a weight matrix and then passed the output through a single sigmoid
function to get the output. Also applied a bias. For the MLP added one hidden layer. Used a
learning rate of .01 (1%) and the loss as Mean Squared Error.

<b>Decision Tree:</b>
Used the ID3 algorithm with a bucket size of 5.
