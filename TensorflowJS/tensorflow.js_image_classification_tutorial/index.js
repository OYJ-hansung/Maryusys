let net;

const NUM_CLASSES = 4;
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();
const learningRate = 0.0001;
const batchSizeFraction = 0.4;
const epochs = 20;
const denseunits = 100;


async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  // net = await mobilenet.load();
  net = await loadTruncatedMobileNet(); //return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
 
  console.log('Successfully loaded model');

  // Create an object from Tensorflow.js data API which could capture image
  // from the web camera as Tensor.
  const webcam = await tf.data.webcam(webcamElement);

  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = async label => {
    // Capture an image from the web camera.
    const img = await webcam.capture();
    const processedImg =
    tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
	const controllerDataset = new ControllerDataset(NUM_CLASSES);
	controllerDataset.addExample(truncatedMobileNet.predict(img), label);
	img.dispose();
  };

  // When clicking a button, add an example for that class.
  // document.getElementById('class-a').addEventListener('click', () => addExample(0));
  // document.getElementById('class-b').addEventListener('click', () => addExample(1));
  // document.getElementById('class-c').addEventListener('click', () => addExample(2));

	  ///////
	document.getElementById('class-a').addEventListener('click', async () => {
	  await tf.nextFrame();
	  await tf.nextFrame();
	  isPredicting = false;
	  train();
	});

	document.getElementById('class-b').addEventListener('click', async () => {
	  await tf.nextFrame();
	  await tf.nextFrame();
	  isPredicting = false;
	  train();
	});

	document.getElementById('class-c').addEventListener('click', async () => {
	  await tf.nextFrame();
	  await tf.nextFrame();
	  isPredicting = false;
	  train();
	});


	/////////


  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();

      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(img, 'conv_preds');
      // Get the most likely class and confidence from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ['A', 'B', 'C'];
      document.getElementById('console').innerText = `
        prediction: ${classes[result.label]}\n
        probability: ${result.confidences[result.label]}
      `;

      // Dispose the tensor to release the memory.
      img.dispose();
    }

    await tf.nextFrame();
  }
}

async function loadTruncatedMobileNet() {

  const mobilenet_origin = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  const saveResult = await mobilenet_origin.save('localstorage://my-model-1');
  const mobilenet = await tf.loadLayersModel('localstorage://my-model-1');

  // const mobilenet = await tf.loadLayersModel('indexeddb://my-model');


  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

class ControllerDataset {
  constructor(numClasses) {
    this.numClasses = numClasses;
  }


  addExample(example, label) {
    // One-hot encode the label.
    const y = tf.tidy(
        () => tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses));

    if (this.xs == null) {
      this.xs = tf.keep(example);
      this.ys = tf.keep(y);
    } else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));

      const oldY = this.ys;
      this.ys = tf.keep(oldY.concat(y, 0));

      oldX.dispose();
      oldY.dispose();
      y.dispose();
    }
  }
}

async function train() {

  const controllerDataset = new ControllerDataset(NUM_CLASSES);
  if (controllerDataset.xs == null) {
    throw new Error('Add some examples before training!');
  }

  // Creates a 2-layer fully connected model. By creating a separate model,
  // rather than adding layers to the mobilenet model, we "freeze" the weights
  // of the mobilenet model, and only train weights from the new model.
  model = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
      tf.layers.flatten(
          {inputShape: truncatedMobileNet.outputs[0].shape.slice(1)}),
      // Layer 1.
      tf.layers.dense({
        units: denseunits,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // Layer 2. The number of units of the last layer should correspond
      // to the number of classes we want to predict.
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  // Creates the optimizers which drives training of the model.
  const optimizer = tf.train.adam(learningRate);
  // We use categoricalCrossentropy which is the loss function we use for
  // categorical classification which measures the error between our predicted
  // probability distribution over classes (probability that an input is of each
  // class), versus the label (100% probability in the true class)>
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

  // We parameterize batch size as a fraction of the entire dataset because the
  // number of examples that are collected depends on how many examples the user
  // collects. This allows us to have a flexible batch size.
  const batchSize =
      Math.floor(controllerDataset.xs.shape[0] * batchSizeFraction);
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: epochs
  });
}


app();