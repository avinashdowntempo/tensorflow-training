import * as tf from '@tensorflow/tfjs';
tf.tensor([1, 2, 3, 4]).print();
tf.scalar(3.14).print();
const x = tf.variable(tf.tensor([1, 2, 3]));
x.assign(tf.tensor([4, 5, 6]));
x.print();

const a = tf.tensor1d([1, 2, 3, 4]);
const b = tf.tensor1d([10, 20, 30, 40]);
a.add(b).print();
tf.add(a, b).print();

const c = tf.tensor2d([1, 2], [1, 2]);
const d = tf.tensor2d([1, 2, 3, 4], [2, 2]);

tf.matMul(c, d).print();
c.matMul(d).print();

const y = tf.tidy(() => {
    // aa, b, and two will be cleaned up when the tidy ends.
    const two = tf.scalar(2);
    const aa = tf.scalar(2);
    const b = aa.square();

    console.log('numTensors (in tidy): ' + tf.memory().numTensors);

    // The value returned inside the tidy function will return
    // through the tidy, in this case to the variable y.
    return b.add(two);
});

const two= tf.scalar(2);
two.print();
two.dispose();


console.log('numTensors (outside tidy): ' + y + tf.memory().numTensors);
y.print();

const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data.
model.fit(xs, ys, {epochs: 10}).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  (model.predict(tf.tensor2d([5], [1, 1])) as tf.Tensor).print()
});