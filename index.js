"use strict";
exports.__esModule = true;
var tf = require("@tensorflow/tfjs");
tf.tensor([1, 2, 3, 4]).print();
tf.scalar(3.14).print();
var x = tf.variable(tf.tensor([1, 2, 3]));
x.assign(tf.tensor([4, 5, 6]));
x.print();
var a = tf.tensor1d([1, 2, 3, 4]);
var b = tf.tensor1d([10, 20, 30, 40]);
a.add(b).print();
tf.add(a, b).print();
var c = tf.tensor2d([1, 2], [1, 2]);
var d = tf.tensor2d([1, 2, 3, 4], [2, 2]);
tf.matMul(c, d).print();
c.matMul(d).print();
var y = tf.tidy(function () {
    // aa, b, and two will be cleaned up when the tidy ends.
    var two = tf.scalar(2);
    var aa = tf.scalar(2);
    var b = aa.square();
    console.log('numTensors (in tidy): ' + tf.memory().numTensors);
    // The value returned inside the tidy function will return
    // through the tidy, in this case to the variable y.
    return b.add(two);
});
var two = tf.scalar(2);
two.print();
two.dispose();
console.log('numTensors (outside tidy): ' + y + tf.memory().numTensors);
y.print();
var model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
// Prepare the model for training: Specify the loss and the optimizer.
model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
// Generate some synthetic data for training.
var xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
var ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);
// Train the model using the data.
model.fit(xs, ys, { epochs: 10 }).then(function () {
    // Use the model to do inference on a data point the model hasn't seen before:
    model.predict(tf.tensor2d([5], [1, 1])).print();
});
