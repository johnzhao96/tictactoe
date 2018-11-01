import tensorflow as tf
import numpy as np
import ttt_game

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape = [1,9], dtype = tf.float32)
W1 = tf.Variable(tf.random_uniform([9,9], 0, 0.01))
Qout = tf.matmul(inputs1, W1)
predict = tf.argmax(Qout, 1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape = [1,9], dtype = tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000

env = ttt_game.GameEnvironment()

#create lists to contain total rewards and steps per episode
record = {-1: 0, 0: 0, 1: 0, -10: 0}
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        done = False
        #The Q-Network
        while not done:
            #Choose an action greedily (with e chance of random action) from the Q-network
            vectorized_board = np.concatenate(s.board)
            vectorized_board.shape = (1,9)
            action, allQ = sess.run([predict, Qout], feed_dict = {inputs1: vectorized_board})
            if np.random.rand(1) < e:
                row, col = env.gameState.getRandomLegalMove_unchecked()
                action[0] = row * 3 + col
            else:
                row = action[0] // 3
                col = action[0] % 3
            
            #Get new state and reward from environment
            s_new, r, done, _ = env.step(row, col)

            #Obtain the Q' values by feeding the new state through our network
            vectorized_board_new = np.concatenate(s_new.board)
            vectorized_board_new.shape = (1,9)
            Q_new = sess.run(Qout, feed_dict = {inputs1: vectorized_board_new})

            #Obtain maxQ' and set our target value for chosen action.
            maxQ_new = np.max(Q_new)
            targetQ = allQ
            targetQ[0, action[0]] = r + y * maxQ_new

            #Train our network using target and predicted Q values
            sess.run([updateModel, W1], feed_dict = {inputs1: vectorized_board, nextQ: targetQ})
            s = s_new
            if done:
                # Record result
                record[r] += 1
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
            
print("Win/Loss/Tie: {}/{}/{}/{}".format(record[1], record[-1], record[0], record[-10]))
print("Win rate: {}".format(record[1] / num_episodes))
