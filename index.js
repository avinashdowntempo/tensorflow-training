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
