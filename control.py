from datetime import datetime, timedelta


class Control:
    def __init__(self, environment, players, is_display=True):
        self.__environment = environment
        self.__players = players
        self.__is_display = is_display
        self.__time_start = datetime.now()
        self.__time_elapsed = timedelta()

    @property
    def time_elapsed(self):
        if self.environment.is_play:
            # プレイ中の場合
            self.__time_elapsed = datetime.now() - self.__time_start

        return self.__time_elapsed

    def play(self, count=1):
        """
        プレイを実施
        :return: 結果
        """

        # 経験(1階層：各プレイヤーのリスト、2階層：各プレイ情報のリスト、3階層：属性のディクショナリ、4階層：データの値)
        experience = list()
        for j in range(len(self.__players)):
            experience.append(list())

        for i in range(count):
            print('start play:{0}回目'.format(i + 1))
            # 各種データの初期化
            self.__time_start = datetime.now()
            self.__environment.start()
            for j in range(len(self.__players)):
                self.__players[j].initialize()
                experience[j].append(dict())
                experience[j][-1]['status_before'] = list()
                experience[j][-1]['action'] = list()
                experience[j][-1]['status_after'] = list()
                experience[j][-1]['reward'] = list()

            if self.__is_display:
                # 表示する場合
                # 初期状態を表示
                self.__environment.display()

            # プレイを最後まで実施
            while True:
                for j in range(len(self.__players)):
                    while True:
                        # 行動前の状態を取得
                        status = self.__environment.status
                        # 行動を取得
                        action = self.__players[j].get_action(self.__environment.status)
                        # 行動を実施
                        can_action = self.__environment.set_action(action)
                        # 環境からの情報を通知
                        reward = self.__players[j].get_reward(status,
                                                              action,
                                                              self.__environment.status,
                                                              self.__environment.is_play,
                                                              self.__environment.score)

                        experience[j][-1]['status_before'].append(status)
                        experience[j][-1]['action'].append(action)
                        experience[j][-1]['status_after'].append(self.__environment.status)
                        experience[j][-1]['reward'].append(reward)

                        if can_action:
                            # 行動が実行できた場合
                            break

                    if self.__is_display:
                        # 表示する場合
                        # 初期状態を表示
                        self.__environment.display()

                    if not self.__environment.is_play:
                        # プレイが終了していた場合
                        # 経過時間の計測を停止
                        self.__time_elapsed = datetime.now() - self.__time_start
                        break

                if not self.__environment.is_play:
                    # プレイが終了していた場合
                    # 経験を調整
                    for j in range(len(self.__players)):
                        self.__players[j].adjust_experience(experience[j][-1], self.__environment.score)
                    break

            # 終了処理を実施
            for j in range(len(self.__players)):
                self.__players[j].finalize(self.__environment.status, self.__environment.score)

        return experience
