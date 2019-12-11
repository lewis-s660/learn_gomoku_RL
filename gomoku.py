import numpy as np


class Gomoku:
    def __init__(self, size=(9, 9)):
        """
        コンストラクタ
        :param size: 盤面サイズ
        """
        self.__board = np.zeros(size)
        self.__is_play = False
        self.__winner = 0
        self.__count = 0
        self.__display_action = ''
        self.__display_result = ''

    @property
    def status(self):
        """ステータス"""
        return self.__board.copy()

    @property
    def is_play(self):
        """
        プレイ中フラグ
        True:プレイ中
        False:非プレイ中
        """
        return self.__is_play

    @property
    def score(self):
        """
        勝者
        1:黒
        0:ドロー
        -1:白
        """
        return self.__winner

    @property
    def count(self):
        """手番"""
        return self.__count

    def start(self):
        """
        プレイを開始する
        :return:　なし
        """
        # 盤面をクリア
        self.__board = np.zeros(self.__board.shape)
        # 勝者をクリア
        self.__winner = 0
        # 手番をクリア
        self.__count = 0
        # 出力文字列をクリア
        self.__display_action = ''
        self.__display_result = ''

        # プレイ中に変更
        self.__is_play = True

    def set_action(self, pos):
        """
        碁石を打つ
        :param pos: 碁石を置く位置
        :return: 配置成功フラグ(True:配置成功、False:配置失敗)
        """
        # 行動実施フラグ
        is_action = False

        # 手番を判定
        if self.__count % 2 == 0:
            # 黒番の場合
            is_black = True
        else:
            # 白番の場合
            is_black = False

        # 碁石を打つ位置を確認
        if (0 <= pos[0]) and (pos[0] < self.__board.shape[0]) \
                and (0 <= pos[1]) and (pos[1] < self.__board.shape[1]):
            # 碁石の打つ位置が盤面内の場合
            if self.__board[pos[0], pos[1]] == 0:
                # 空のマスに打つ場合
                string = '第{0}手 '.format(self.__count + 1)
                if is_black:
                    # 黒の手番の場合
                    self.__board[pos[0], pos[1]] = 1
                    string += '黒'
                else:
                    # 白の手番の場合
                    self.__board[pos[0], pos[1]] = -1
                    string += '白'

                self.__display_action = string + ':({0}, {1})'.format(pos[0], pos[1])
                self.__count += 1

                self.__judge(pos)

                is_action = True

        return is_action

    def __judge(self, pos):
        """
        勝敗を判定
        :param pos: 碁石を打つ位置
        :return:　なし
        """
        val = self.__board[pos[0], pos[1]]
        axes = list([[[0, 1], [0, -1]],
                      [[1, 0], [-1, 0]],
                      [[1, 1], [-1, -1]],
                      [[1, -1], [-1, 1]]])

        for axis in axes:
            # 探索軸分ループ
            count = 1
            for i in range(len(axis)):
                # 探索軸ごとの探索方向分ループ
                # 開始位置を取得
                pos_search = pos.copy()
                while True:
                    # 次の探索マスを取得
                    pos_search += axis[i]
                    if (0 <= pos_search[0]) and (pos_search[0] < self.__board.shape[0]) \
                            and (0 <= pos_search[1]) and (pos_search[1] < self.__board.shape[1]):
                        # 次の探索マスが盤面上にある場合
                        if val == self.__board[pos_search[0], pos_search[1]]:
                            # 探索マスの値が同色の場合
                            count += 1
                        else:
                            # 探索マスの値が同色でないまたは空の場合
                            break
                    else:
                        # 次の探索マスが盤面上にない場合
                        break

            if 5 <= count:
                # 勝者が決った場合
                # 勝者を設定
                self.__winner = val
                # プレイを終了
                self.__is_play = False
                # 終了情報を設定
                self.__display_result = '第{0}手 {1}の勝利です。'.format(self.__count, '黒' if self.__winner == 1 else '白')
                break

        # ドロー判定処理
        if self.__is_play and (val == -1) \
                and (((self.__board.shape[0] * self.__board.shape[1]) - ((self.__count // 2) * 2)) <= 1):
            # プレイ中かつ白手番かつ次の白手番が回ってこない場合
            # ドローを設定
            self.__winner = 0
            # プレイを終了
            self.__is_play = False
            # 終了情報を出力
            self.__display_result = '第{0}手 引き分けです。'.format(self.__count)

    def display(self):
        """
        表示出力
        :return: なし
        """

        output = ''
        for i in range(self.__board.shape[0]):
            for j in range(self.__board.shape[1]):
                if self.__board[i, j] == 1:
                    output += '○'
                elif self.__board[i, j] == -1:
                    output += '●'
                else:
                    output += '+'
                output += '  '
            output += '\r\n'

        print(self.__display_action)
        print(output + self.__display_result)
