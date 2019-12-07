# Neural Network Weight Optimization With Eagle Strategy :
The traditional backprogation algorithm used in weight optimization in neural networks uses gradient descent, i.e., it calculates the gradient of the error function with respect to the neural network's weights, and adjusts the network weights accordingly in every epoch. The backpropagation method achieves good accuracy, but we observed that improvements can be made on the training time of the model. We propose an approach that utilizes eagle strategy for weight optimization to improve the training time of a neural network. Our approach utilises a method known as Lévy flight that performs a global search on weights, and then local search is performed on the neighbours of the weights that perform better than a given threshold accuracy. We utilise simulated annealing and random hill climbing to perform local search on the weights that pass the threshold. Our results show that training neural networks in this form achieves faster convergence.

# Running the program :
1. Type $pip install -r requirements.txt in your computer's terminal.
2. Type $python3 AI.py. The program compares the training time and accuracy of three neural network weight optimization      techniques.
3. AI_test.py was used to plot graphs related to the results.
