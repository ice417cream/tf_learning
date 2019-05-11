import tensorflow as tf
import numpy as np
import time
import threading

step = 0

x = tf.random_uniform((5000, 5000), dtype=tf.float32)

y = tf.matmul(x,x)


class Worker(object):
    def __init__(self,id):
        self.id = id
    def work(self):
        global step
        while not coord.should_stop() and step<100:
            for i in range(1):
                sess.run(y)
            print(step,self.id)

            step += 1


if __name__ =="__main__":
    sess=tf.Session()
    with tf.device("/cpu:0"):
        workers = []
        for i in range(1):
            workers.append(Worker(i))

    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())

    time_start = time.time()
    worker_threads = []
    for worker in workers:
        job = lambda:worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)
    print(time.time()-time_start)
