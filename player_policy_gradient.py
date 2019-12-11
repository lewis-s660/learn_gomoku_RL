import os
import math

import numpy as np
import tensorflow.compat.v1 as tf
import tensorlayer as tl


class PlayerPolicyGradient:
    def __init__(self, is_black, size_board, path_weights, name, mode='', rate_learning=0.01, rate_decay_reward=0.9):
        self.__is_black = is_black
        self.__size_board = size_board
        self.__path_weights = path_weights
        self.__name = name
        self.__mode = mode
        self.__rate_learning = rate_learning
        self.__rate_decay__reward = rate_decay_reward
        self.__last_step = 0

        self.__create_model()

        self.__reward = {'win': 10, 'lose': -10, 'draw': 1, 'invalid': -1, 'else': 0}

    def initialize(self):
        pass

    def finalize(self, status, score):
        pass

    def __create_model(self):
        """
        モデルを生成
        """
        number_node_layers = [32, 64, 128, 1024, 1]

        self.__graph = tf.Graph()
        with self.__graph.as_default():
            with tf.name_scope('inputX'):
                global_step = tf.Variable(0, name='global_step', trainable=False)
                board = tf.placeholder(dtype=tf.float64, shape=[None, self.__size_board[0], self.__size_board[1], 3])
                #x = tf.reshape(board, [-1, self.__size_board[0] * self.__size_board[1] * 3])

            with tf.name_scope('convolution_layer_1'):
                filter_1 = tf.Variable(np.random.randn(3, 3, 3, number_node_layers[0]),
                                       name='filter_1')
                convolution_1 = tf.nn.conv2d(board, filter=filter_1, strides=[1, 1, 1, 1], padding='SAME')
                b_1 = tf.Variable(np.zeros(number_node_layers[0]), name='b_1')
                out_1 = tf.nn.relu(convolution_1 + b_1)

            with tf.name_scope('max_pooling_layer_2'):
                out_2 = tf.nn.max_pool(out_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            with tf.name_scope('convolution_layer_3'):
                filter_2 = tf.Variable(np.random.randn(3, 3, number_node_layers[0], number_node_layers[1]),
                                       name='filter_2')
                convolution_2 = tf.nn.conv2d(out_2, filter=filter_2, strides=[1, 1, 1, 1], padding='SAME')
                b_2 = tf.Variable(np.zeros([number_node_layers[1]]))
                out_3 = tf.nn.relu(convolution_2 + b_2)

            with tf.name_scope('max_pooling_layer_4'):
                out_4 = tf.nn.max_pool(out_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            with tf.name_scope('flatten_layer_5'):
                height = math.ceil(math.ceil(self.__size_board[0] / 2) / 2)
                width = math.ceil(math.ceil(self.__size_board[1] / 2) / 2)
                size_flat = height * width * number_node_layers[1]
                out_5 = tf.reshape(out_4, [-1, size_flat])

            with tf.name_scope('full_connection_layer_6'):
                size_board = self.__size_board[0] * self.__size_board[1]
                weight = tf.Variable(np.random.randn(size_flat, size_board), name='weight')
                b_3 = tf.Variable(np.zeros(size_board), name='b_3')
                y_hat = tf.matmul(out_5, weight) + b_3

            with tf.name_scope('probability'):
                probability = tf.nn.softmax(y_hat, axis=1)

            with tf.name_scope('loss'):
                action = tf.placeholder(dtype=tf.int32, shape=[None])
                reward = tf.placeholder(dtype=tf.float64, shape=[None])

                loss = tl.rein.cross_entropy_reward_loss(y_hat, action, reward)

            with tf.name_scope('train'):
                train = tf.train.AdamOptimizer().minimize(loss)

            saver = tf.train.Saver()

        self.__tensor = {'global_step': global_step,
                         'board': board,
                         'probability': probability,
                         'action': action,
                         'reward': reward,
                         'loss': loss,
                         'train': train,
                         'saver': saver}

    def get_action(self, board):
        """
        次に配置する碁石の位置を取得
        :param board: 盤面
        :return: 碁石の位置
        """

        if self.__mode == 'user':
            # ユーザーモード
            while True:
                print('{0}の手番です。碁石を置く位置を入力してください。'.format('黒' if self.__is_black else '白'))
                x = input('横方向の位置：')
                y = input('縦方向の位置：')
                try:
                    pos = np.array([int(x), int(y)])
                    break
                except:
                    pass
        elif self.__mode == 'random':
            # ランダムモード
            area = list()
            for i in range(board.shape[0]):
                for j in range(board.shape[1]):
                    if board[i, j] == 0:
                        area.append(np.array([i, j]))

            pos = area[np.random.randint(0, len(area))]
        elif self.__mode == 'random2':
            # ランダム2モード
            x = np.random.randint(0, board.shape[1])
            y = np.random.randint(0, board.shape[0])

            pos = np.array([x, y])
        else:
            # 盤面を整形
            boards = self.__form_board(board)

            with tf.Session(graph=self.__graph) as sess:
                self.__restore_weights(sess)
                # 次に配置する碁石の位置を決定
                probability = sess.run(self.__tensor['probability'], feed_dict={self.__tensor['board']: boards})
                probability = (np.array(probability)).reshape([-1])

            size = len(probability)
            # 空マスを取得
            while True:
                index = np.random.choice(range(size), p=probability)
                pos = (index // board.shape[1], index % board.shape[1])

                if board[pos[0], pos[1]] == 0:
                    # 空マスの場合
                    break
                else:
                    probability = np.delete(probability, index, 0)
                    probability = np.exp(probability) / np.sum(np.exp(probability))
                    size -= 1

        return np.array(pos)

    def get_reward(self, board_before, pos, board, is_play, winner):
        """
        報酬を取得
        :param board_before: 行動前の状態
        :param pos: 行動
        :param board: 行動後の状態
        :param is_play: プレイ中フラグ
        :param winner: 勝者
        :return: 報酬
        """
        # プレイ中の場合
        if (board_before == board).all():
            # 状態が変化していない場合
            reward = self.__reward['invalid']
        else:
            # 状態が変化している場合
            reward = self.__reward['else']

        return reward

    def adjust_experience(self, experience, winner):
        """
        経験の情報を調整
        :param experience: 経験
        :param winner: 勝者
        :return: なし
        """
        if winner == 0:
            # ドローの場合
            experience['reward'][-1] = self.__reward['draw']
        elif winner == (1 if self.__is_black else -1):
            # 自身が勝った場合
            experience['reward'][-1] = self.__reward['win']
        else:
            # 自身が負けた場合
            experience['reward'][-1] = self.__reward['lose']

    def fit(self, experience, epochs=100, size_batch=20):
        status = list()
        action = list()
        reward = list()
        for i in range(len(experience)):
            status += experience[i]['status_before']
            action += experience[i]['action']
            reward += experience[i]['reward']

        # 行動のフォーマットを変更((x, y) => x * 碁盤幅 + y)
        action = np.array(action)
        action = action[:, 0] * len(status[0][0]) + action[:, 1]

        # ステータスのフォーマット変換
        for i in range(len(status)):
            status[i] = self.__form_board(status[i], False)

        count = (len(status) // size_batch) + 1

        with tf.Session(graph=self.__graph) as sess:
            self.__restore_weights(sess)
            loss_history = list()

            for i in range(epochs):
                # 経験シャッフル用インデックスを生成
                index = np.random.choice(range(len(status)), len(status), replace=False)
                losses = list()
                for j in range(count):
                    if action.shape[0] <= j * size_batch:
                        # 1つもデータがない場合
                        break
                    loss, _ = sess.run([self.__tensor['loss'], self.__tensor['train']],
                                       feed_dict={self.__tensor['board']: status[j * size_batch: (j + 1) * size_batch],
                                                  self.__tensor['action']: action[j * size_batch: (j + 1) * size_batch],
                                                  self.__tensor['reward']: reward[j * size_batch: (j + 1) * size_batch]})

                    losses.append(loss)

                loss_history.append(np.mean(losses))
                print('     color:{0} epochs:{1}  loss:{2:.3f}'.format('black' if self.__is_black else 'white', i, loss_history[-1]))

            self.__save_weights(sess)

    def __restore_weights(self, sess):
        """
        モデルの重みを復元するメソッド
        :param sess:　セッション
        """
        checkpoint = tf.train.get_checkpoint_state(self.__path_weights)
        try:
            if checkpoint:
                last_model = checkpoint.model_checkpoint_path
                self.__tensor['saver'].restore(sess, last_model)
                self.__last_step = sess.run(self.__tensor['global_step'])
            else:
                sess.run(tf.global_variables_initializer())
                self.__last_step = 0
        except:
            sess.run(tf.global_variables_initializer())
            self.__last_step = 0

    def __save_weights(self, sess):
        """
        モデルの重みを保存するメソッド
        :param sess: セッション
        :return: なし
        """
        self.__tensor['saver'].save(sess,
                                    os.path.join(self.__path_weights, self.__name),
                                    global_step=self.__last_step,
                                    write_meta_graph=False)

    def __form_board(self, board, add_axis=True):
        """
        盤面整形
        :param board: 整形前盤面
        :return: 整形後盤面
        """
        if self.__is_black:
            # 自身が黒の場合
            own = 1
            other = -1
        else:
            # 自身が白の場合
            own = -1
            other = -1

        boards = list()
        boards.append((board == own).astype(np.int))
        boards.append((board == other).astype(np.int))
        boards.append((board == 0).astype(np.int))
        boards = (np.array(boards)).transpose([1, 2, 0])
        if add_axis:
            boards = boards[np.newaxis, :, :, :]

        return boards
