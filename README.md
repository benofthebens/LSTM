# LSTM
This is part of RedPanda computer science project
it is the most messiest code but i think it works
## Errors
- Only error forgot to change to the output weight variables
- Learning Rate too high 
- forgot when adding the dataset had to update weights not after every single one was complete but during after the training
- csv writer/reader execution
- wrong attributes used for backprogation

## First Iteration of the model
- just the basic implimenation of the lstm model with no loops to test whether or not it works 
- uses the csv writer to write to the csv the weights
- Unfinished and broken backward propgation through time - this is because of the incorectly defined attibutes inside of the code  if i were to change them it should all work fine
- when i do the next iteration i'll have to put the writing to csv inside of the training loop because it has to store for every iteration
- change output parameater to layers[i].outputLayer or HiddenState because of having to calculate change in cost in respect to the hidden state for that spesific layer rather than teh final output
- make learning rate a variable
- back propgation seems to work just the layout and logic of my code that needs a tweak apart from the incorrect paramaters and attibutes used

## Second iteration 
- completed all of the changes above whihc means now it is traianble and it works
- but the structure of programming needs to be improved
## Third iteration 
essentialy coded it in a more managable way
### errors 
- forgot the get set values unable to read from csv 
- needed to set to false to write over the data
- added for loop that wasnt suposed to be there
- this caused records to be repeated 
