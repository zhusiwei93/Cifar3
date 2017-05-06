from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from IPython.core.debugger import Tracer

import numpy as np
import tensorflow as tf

import cifar3

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar3_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'train_eval',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar3_train',
                           """Directory where to read model checkpoints.""")

def evaluate():

    with tf.Graph().as_default() as g:

        eval_data = FLAGS.eval_data == 'test'
        images, labels = cifar3.inputs(eval_data=eval_data)
        logits = cifar3.inference(images)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar3.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:

            # restore from checkpoint
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return
            
            Tracer()()
            
            print('Start predicting...')
            predictions = sess.run([logits])
            print(predictions)
            print('End prediction')

def main(argv=None):
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
        tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()

if __name__ == '__main__':
    tf.app.run()
