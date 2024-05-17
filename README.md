# DL_A3
#####################################################################################################################
building and training the model for aksharantar/telugu dataset(start, end, pad tokens: <,>,? respectiveley): 

DL_A3 consist of 3 sets files

I. cs23m005-a3.ipynb, cs23m005-a3.py : without attention model
II. cs23m005-a3-attn.ipynb, cs23m005-a3-attn.py : with attention model
III. /prediction_attention/prediction_report.csv : contains the predicted output for the test data

#####################################################################################################################
I. without attention: cs23m005-a3.py

run cs23m005-a3.py with the optinal arguments by using the wandb API key.

arguments are as follows

'-wp', '--wandb_project', type=str, default="Assignment3", help="wandb project name"

'-opt', '--optimizer', type=str, default="nadam", choices = ['adam','nadam'], help="optimizer for backprop"

'-bS', '--batch_size', type=int, default=32, choices = [32, 64, 128, 256], help="batch size"

'-nE', '--num_epochs', type=int, default=25, choices = [5, 10], help="number of epochs"

'-lR', '--learning_rate', type=float, default=1e-3, choices = [1e-3, 1e-4], help="learning rate"

'-hS', '--hidden_size', type=int, default=128, choices = [32, 64, 128, 256], help="hidden size"

'-eS', '--embedding_size', type=int, default=256, choices = [32, 64, 128, 256], help="embedding size"

'-nL', '--num_layers', type=int, default=3, choices = [1,2,3,4,5], help="number of layers"

'-cell', '--cell_type', type=str, default="LSTM", choices = ['LSTM', 'GRU', 'RNN'], help="cell type"

'-dr', '--dropout_in', type=float, default=0.2, choices = [0,0.2,0.3,0.5], help="drop out"


This initiates a call to train() which will build the seq2seq model and train it as per the requirement, this will produce validation and test accuracy and logs it in wandb CS23M005/Assignment3.

Detailed description of the functions:

parse_args()

This function takes the input from the command line arguments and call train() with required parameters.

This function has default values in which case if user does not provide arguments

**train()** function does the following

1. **Initialize Dataloaders**:
   - `train_dataloader` for training data.
   - `valid_dataloader` for validation data.

2. **Set Up Model Components**:
   - Create the encoder and decoder networks.
   - Combine them into a Seq2Seq model.

3. **Define Loss Function and Optimizer**:
   - Use `nn.NLLLoss` for the loss function.
   - Choose between Adam and NAdam optimizers.

4. **Training Loop**:
   - Iterate over the specified number of epochs.
   - For each epoch:
     - Set model to training mode.
     - Loop over batches in the training dataloader.
     - Move input and target data to the device.
     - Perform forward pass to get model output.
     - Calculate predictions.
     - Zero gradients, compute loss, and backpropagate.
     - Update model parameters.

5. **Track Metrics**:
   - Calculate total loss and accuracy for training data.
   - Compute validation loss and accuracy after each epoch.
   - Log metrics using Weights and Biases (wandb).
     
### Encoder Class:

1. **Initialization**:
   - The `Encoder` class initializes the encoder part of the Seq2Seq model.
   - It accepts parameters such as input size, embedding size, hidden size, number of layers, cell type, batch size, and dropout rate.
   - It sets up the character embedding layer and the chosen recurrent neural network (RNN, LSTM, or GRU) based on the cell type.

2. **Forward Method**:
   - The `forward` method takes input data `x` and hidden state `hidden` as inputs.
   - It passes the input data through the embedding layer and then through the recurrent layer (RNN, LSTM, or GRU).
   - The output and hidden state from the recurrent layer are returned.

3. **Initial State**:
   - The `getInitialState` method initializes the initial hidden state of the encoder.
   - It returns a tensor of zeros with the shape `(num_layers, batch_size, hidden_size)`.

### Decoder Class:

1. **Initialization**:
   - The `Decoder` class initializes the decoder part of the Seq2Seq model.
   - It accepts parameters similar to the encoder but also includes the output size (size of the vocabulary) for generating predictions.
   - It sets up the Devanagari character embedding layer, the recurrent layer, a fully connected layer, and a softmax layer.

2. **Forward Method**:
   - The `forward` method takes input data `x` and hidden state `hidden` as inputs.
   - It passes the input data through the embedding layer and then through the recurrent layer.
   - The output from the recurrent layer is passed through a fully connected layer and then through a softmax layer to generate predictions.
   - Predictions and the updated hidden state are returned.

3. **Output Prediction**:
   - The `forward` method returns predictions for each time step in the decoder's output sequence.
   - The output tensor has a shape of `(target_length, batch_size, output_size)`.

### Seq2Seq Class:

1. **Initialization**:
   - The `Seq2Seq` class initializes the Seq2Seq model by connecting an encoder and a decoder.
   - It accepts instances of encoder and decoder classes as inputs.

2. **Forward Method**:
   - The `forward` method takes source data (`source`) and target data (`target`) as inputs.
   - It iterates over each time step in the target sequence, feeding the input at each time step into the decoder.
   - Teacher forcing is optionally applied based on the specified `teacher_forcing_ratio`.
   - The method returns the outputs of the decoder for each time step.

3. **Teacher Forcing**:
   - Teacher forcing is a technique where instead of using the decoder's output from the previous time step as input for the next time step, the ground truth target token is used.
   - The `forward` method implements teacher forcing with a specified probability (`teacher_forcing_ratio`).

#####################################################################################################################

II. with attention: cs23m005-a3-attn.py:
This is same as cs23m005-a3.py. follow the same steps as mentioned above. The difference between with attention and without attention is highlighted below

attention calculation is the main difference. This is done after the encoder module and for each input of decoder

### Differences in New Classes:

1. **Attention Mechanism**:
   - The new classes (`Attention`, `Decoder`) include an attention mechanism, which was not present in the earlier classes.
   - The `Attention` class computes attention scores and weights based on query and keys, enhancing the model's ability to focus on relevant parts of the input sequence during decoding.

2. **Updated Decoder Architecture**:
   - The `Decoder` class in the new implementation integrates the attention mechanism into its architecture.
   - It concatenates the current embedding with the context vector from the attention mechanism before passing it through the recurrent layer.
   - This enables the decoder to attend to different parts of the input sequence dynamically during decoding.

3. **Enhanced Decoder Output**:
   - The decoder's forward method now returns not only predictions but also attention weights for visualization and analysis.
   - This allows for better understanding of the model's decision-making process and visualization of where the model is focusing its attention during decoding.

4. **Improved Seq2Seq Model**:
   - The `Seq2Seq` class has been updated to accommodate the changes in the encoder and decoder classes.
   - It now passes the encoder's final layer states to the decoder along with the input sequence, facilitating the attention mechanism's operation.

5. **Updated Forward Pass**:
   - During the forward pass of the `Seq2Seq` model, attention weights are computed and stored for each time step.
   - This enables the model to capture the alignment between input and output sequences, aiding in tasks such as machine translation or text summarization.
   ######################################################################################################################
   
