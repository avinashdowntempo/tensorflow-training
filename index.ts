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