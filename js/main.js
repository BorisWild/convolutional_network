const status = document.getElementById('status');
if (status) {
  status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
}
//Note that the data is already in normalized form.
// Since grayscale images can only contain numbers between 0 and 255,
// dividing each value by 255 will normalize the data between 0 and 1 if you had to do this manually.
import {TRAINING_DATA} from './mnist.js';


// Grab a reference to the MNIST input values (pixel data).
const INPUTS = TRAINING_DATA.inputs;
// Grab reference to the MNIST output values.
const OUTPUTS = TRAINING_DATA.outputs;

// Shuffle the two arrays in the same way so inputs still match outputs indexes.
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// The input tensor is a 2-Dimensional Tensor since it is an array of grayscale images,
// each of which is an array of 784 values. ( [ [1, 2],[1, 3],...[...] ] )
const INPUTS_TENSOR = tf.tensor2d(INPUTS);

// Output feature Array is 1 dimensional.
//The ‘tf.oneHot’ function takes two parameters,
// the tensor that needs to be encoded and the number of classes that represent the data.
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), 10);

// Now actually create and define model architecture.
//define model... activate layer by layer...first output will be input for another layer
const model = tf.sequential();

//Add the input layer. Since the input images are 28x28 pixels in size, the input shape is [784]
//Set the number of neurons to 32 and add the ‘reLU’ activation function.
//Note that the number of neurons is a parameter that an ML engineer arrives at
// after experimentation keeping in mind the balance between accuracy and speed
model.add(tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}));

//Add a hidden layer with 16 neurons with the ‘relu’ activation function
model.add(tf.layers.dense({units: 16, activation: 'relu'}));

//Add the final output layer with 10 neurons and the ‘softmax’ activation function.
//This ensures all outputs add up to one, essentially providing percentage confidences for each output.
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

//Observe that the model has around 26,000 trainable parameters.
//Working with pixel data can lead to very large models.
model.summary();

train();

const PREDICTION_ELEMENT = document.getElementById('prediction');
const CANVAS = document.getElementById('canvas');
const CTX = CANVAS.getContext('2d');

//TensorFlow.js also has a tf.browser.toPixels() function that you can use if you wish to stay completely in Tensor land. A very useful function if you have all the values in a Tensor of the right shape for a given image and you want to push those values to a HTML canvas in one line of code.
function drawImage(digit) {
  digit = tf.tensor(digit, [28, 28]);
  tf.browser.toPixels(digit, CANVAS);
  // Perform a new classification after a certain interval.
  setTimeout(evaluate, 2000);
}

/*function drawImage(digit) {
  //you can access the current canvas data using CTX.getImageData.
  //You can pass the starting position of the canvas to this function,
  //which is 0,0 (the top left corner of the canvas).
  var imageData = CTX.getImageData(0, 0, 28, 28);

  //Here you set the Red Green and Blue channels to simply be the input images value multiplied by 255.
  //Remember your input images were in normalized form for training the model,
  //so to get them into regular color values you need to multiply by 255.
  for (let i = 0; i < digit.length; i++) {
    imageData.data[i * 4] = digit[i] * 255;      // Red Channel.
    imageData.data[i * 4 + 1] = digit[i] * 255;  // Green Channel.
    imageData.data[i * 4 + 2] = digit[i] * 255;  // Blue Channel.
    imageData.data[i * 4 + 3] = 255;             // Alpha Channel.
  }

  // Render the updated array of data to the canvas itself.
  CTX.putImageData(imageData, 0, 0);

  // Perform a new classification after a certain interval.
  setTimeout(evaluate, 2000);
}*/

function evaluate() {
  const OFFSET = Math.floor((Math.random() * INPUTS.length)); // Select random from all example inputs.

  let answer = tf.tidy(function () {
    //Use the ‘model.predict’ function to perform a prediction on your new input.
    //Remember, the ‘model.predict’ function expects a batch of images as input.
    //To avoid an error, call expandDims on the tensor 1d you just created
    //to change it to a tensor2 with just one value to avoid errors
    let newInput = tf.tensor1d(INPUTS[OFFSET]).expandDims();
    let output = model.predict(newInput);

    //classes
    output.print();
    //Before you use argMax, you need to use the ‘squeeze’ function first
    //to convert the 2d output tensor into a 1d tensor
    return output.squeeze().argMax();
  })

  answer.array().then(function(index) {
    PREDICTION_ELEMENT.innerText = index;
    PREDICTION_ELEMENT.setAttribute('class', (index === OUTPUTS[OFFSET]) ? 'correct' : 'wrong');
    answer.dispose();
    drawImage(INPUTS[OFFSET]);
  });

}

async function train() {
  // Compile the model with the defined optimizer and specify our loss function to use.
  model.compile({
    //You can use the stochastic gradient descent algorithm instead of ‘adam’,
    //but adam has the added advantage where
    //it automatically changes the learning rate over time for you
    //often finding slightly better values for weights and biases
    //thus reducing the loss further and well suited for image problems like this.
    optimizer: 'adam',
    loss: 'categoricalCrossentropy', //categoricalCrossentropy was used instead of meanSquaredError
    //This contains an array of extra metrics that can be logged.
    //In this case, log the metric ‘accuracy’ as a measure of how many images
    // are predicted correctly from the training data.
    metrics: ['accuracy']
  });

  let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
    shuffle: true,        // Ensure data is shuffled again before using each epoch.
    validationSplit: 0.2, //Set validationSplit to 0.2 (20%).
    //Note that you can experiment with batch, epochs values to get the best speed and accuracy!
    batchSize: 512,       // Update weights after every 512 examples.
    epochs: 50,           // Go over the data 50 times!
    callbacks: {onEpochEnd: logProgress}
  });

  //clear
  OUTPUTS_TENSOR.dispose();
  INPUTS_TENSOR.dispose();

  evaluate(); // Once trained we can evaluate the model.
}

function logProgress(epoch, logs){
  console.log('Data for epoch '+epoch, Math.sqrt(logs.loss), logs.acc)
}
