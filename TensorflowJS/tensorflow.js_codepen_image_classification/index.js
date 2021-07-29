// console.clear();

const { of, fromEvent, merge, Scheduler, interval } = rxjs;
const { flatMap, takeUntil, map, switchMap } = rxjs.operators;

const NUM_CLASSES = 4;
const MODEL_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

const loadMobilenet = async function (url) {
  const mobilenet = await tf.loadModel(url);
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({
    inputs: mobilenet.input,
    outputs: layer.output
  })
};

const cropImage = (image) => {
  const [height, width] = image.shape;
  const size = Math.min(width, height);
  const start = [(height - size) / 2, (width - size) / 2, 0];
  const end = [size, size, 3];
  return image.slice(start, end);
}

const capture = (webcam) => tf.tidy(() => {
  const webcamImage = tf.fromPixels(webcam);
  const cropped = cropImage(webcamImage);
  const expanded = cropped.expandDims();
  // normalize
  return expanded.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
});


let mobilenet;
let webcamera;
let model;
const examples = {
  xs: null,
  ys: null
};

const learningRate = 0.0001;
const batchSizeFraction = 0.4;
const epochs = 20;
const hiddenUnits = 100;

const addExample = (example, label) => {
  const y = tf.tidy(() => {
    return tf.oneHot(tf.tensor1d([label]).toInt(), NUM_CLASSES);
  });

  if (examples.xs === null) {
    examples.xs = tf.keep(example);
    examples.ys = tf.keep(y);
  } else {
    const oldX = examples.xs;
    const oldY = examples.ys;

    examples.xs = tf.keep(oldX.concat(example));
    examples.ys = tf.keep(oldY.concat(y));

    oldX.dispose();
    oldY.dispose();
    y.dispose();
  }
};

const train = () => {

  if (examples.xs === null) {
    throw new Error('Add some examples before training!');
  }

  model = tf.sequential({
    layers: [
      tf.layers.flatten({ inputShape: [7, 7, 256] }),

      tf.layers.dense({
        units: hiddenUnits,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),

      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  const optimizer = tf.train.adam(learningRate);

  model.compile({ optimizer, loss: 'categoricalCrossentropy' });

  const batchSize = Math.floor(examples.xs.shape[0] * batchSizeFraction);
  if (batchSize <= 0) {
    throw new Error('Batch size is 0 or NaN.');
  }

  model.fit(examples.xs, examples.ys, {
    batchSize,
    epochs,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        console.log(`Loss: ${logs.loss.toFixed(5)}`);
        await tf.nextFrame();
      }
    }
  });
};

const predict = async () => {
  const predicted = tf.tidy(() => {
    const img = capture(webcamera);
    const activation = mobilenet.predict(img);
    const predictions = model.predict(activation);
    return predictions.as1D().argMax();
  });

  const classid = (await predicted.data())[0];

  predicted.dispose();

  await tf.nextFrame();
  
  return classid;
};

const setupWebcamera = async (webcam) => {
  webcam.addEventListener('loadeddata', async () => {
    const { videoWidth, videoHeight } = webcam;
    const aspectRatio = videoWidth / videoHeight;

    if (videoWidth < videoHeight) {
      webcam.height = webcam.width / aspectRatio;
    } else {
      webcam.width = aspectRatio * webcam.height;
    }
  });

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    webcam.srcObject = stream;

  } catch (err) {
    console.error(err);
  }
};

const pressing = el => fromEvent(el, 'mousedown')
  .pipe(
    switchMap(_ =>
      interval(0, Scheduler.animationFrame).pipe(
        takeUntil(fromEvent(el, 'mouseup'))
      )
    )
  );

const setupUI = () => {
  const dirs = {
    up: { label: 0, text: '☝️' },
    left: { label: 1, text: '👈' },
    down: { label: 2, text: '👇' },
    right: { label: 3, text: '👉' },
  };
  
  const joyUp = document.querySelector('.joystick .up');
  const joyLeft = document.querySelector('.joystick .left');
  const joyDown = document.querySelector('.joystick .down');
  const joyRight = document.querySelector('.joystick .right');

  const startTrain = document.querySelector('.train');
  const startPredict = document.querySelector('.predict');

  const predicted = document.querySelector('.predicted');
  
  const upClick = pressing(joyUp)
    .pipe(map(_ => dirs.up.label));
  const leftClick = pressing(joyLeft)
    .pipe(map(_ => dirs.left.label));
  const downClick = pressing(joyDown)
    .pipe(map(_ => dirs.down.label));
  const rightClick = pressing(joyRight)
    .pipe(map(_ => dirs.right.label));

  merge(upClick, leftClick, rightClick, downClick)
    .subscribe(label => {
      if (webcamera) {
        const img = capture(webcamera);
        const example = mobilenet.predict(img);
        addExample(example, label);
      }
    });

  fromEvent(startTrain, 'click').subscribe(async () => {
    await tf.nextFrame();
    train(examples);
  });

  fromEvent(startPredict, 'click').pipe(
    switchMap(_ =>
      interval(0, Scheduler.animationFrame)
    )
  )
  .subscribe(async () => {
    const label = await predict();
    const key = Object.keys(dirs).find(k => dirs[k].label === label);
    predicted.textContent = `${dirs[key].text}`;
  });
};

(async function () {
  setupUI();

  mobilenet = await loadMobilenet(MODEL_URL);

  webcamera = document.querySelector('#webcam');

  await setupWebcamera(webcamera);
})();