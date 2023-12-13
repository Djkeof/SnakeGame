from asyncio import current_task
import math
import random
from threading import currentThread
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time


class SnakeGameClass:
    def __init__(self, path_food, path_snake_head):
        # 初始化游戏
        self.detect_collision = True  # 默认开启碰撞检测

        self.points = []  # 蛇的所有节点
        self.lengths = []  # 每个节点间距离
        self.current_length = 0  # 蛇当前长度
        self.allowed_length = 150  # 蛇最大长度
        self.previous_hand = (0, 0)  # 上一个节点

        # 读取蛇头和食物图片
        self.img_snake_head = cv2.imread(path_snake_head, cv2.IMREAD_UNCHANGED)
        self.img_snake_head = cv2.cvtColor(self.img_snake_head, cv2.COLOR_BGRA2BGR)
        self.img_snake_head = cv2.resize(self.img_snake_head, (50, 50))  # 调整蛇头图片的大小
        self.h_snake_head, self.w_snake_head, _ = self.img_snake_head.shape  # 获取蛇头图片的高度和宽度

        self.img_food = cv2.imread(path_food, cv2.IMREAD_UNCHANGED)
        self.h_food, self.w_food, _ = self.img_food.shape
        self.food_point = 0, 0
        self.random_food_location()

        # 游戏参数
        self.score = 0
        self.game_over = False

    def reset_game(self):
        # 重置游戏状态
        self.points = []
        self.lengths = []
        self.current_length = 0
        self.allowed_length = 150
        self.previous_hand = (0, 0)
        self.random_food_location()

    def check_collision(self, img_main, cx, cy):
        # 检查碰撞
        min_dist = 0  # 初始化 min_dist
        if self.detect_collision:
            if self.points:
                pts = np.array(self.points[:-4], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img_main, [pts], False, (0, 200, 0), 3)
                min_dist = cv2.pointPolygonTest(pts, (cx, cy), True)

            if -1 <= min_dist <= 1:
                print("Game Over")
                self.game_over = True
                self.reset_game()

    def draw_snake(self, img_main):
        # 画蛇身
        if self.points:
            body_pts = np.array(self.points[:-2], np.int32)
            body_pts = body_pts.reshape((-1, 1, 2))
            cv2.polylines(img_main, [body_pts], False, (0, 0, 255), 20)
            cx, cy = self.points[-1]
            cv2.circle(img_main, (cx, cy), 20, (200, 0, 200), cv2.FILLED)
            self.draw_snake_head(img_main, cx, cy)

    def draw_snake_head(self, img_main, cx, cy):
        # 画蛇头
        start_x = max(0, cx - self.w_snake_head // 2)
        end_x = min(img_main.shape[1], start_x + self.img_snake_head.shape[1])
        start_y = max(0, cy - self.h_snake_head // 2)
        end_y = min(img_main.shape[0], start_y + self.img_snake_head.shape[0])

        img_main[start_y:end_y, start_x:end_x] = self.img_snake_head[:end_y - start_y, :end_x - start_x]

    def random_food_location(self):
        # 随机生成食物位置
        self.food_point = random.randint(100, 1000), random.randint(100, 600)

    def update(self, img_main, current_hand):
        # 游戏逻辑更新

        if self.game_over:
            # 游戏结束时显示分数
            cvzone.putTextRect(img_main, "Game Over", (300, 400), scale=5, thickness=10, offset=20)
            cvzone.putTextRect(img_main, f'Score: {self.score}', (300, 550), scale=5, thickness=10, offset=20)
        else:
            # 游戏进行中

            px, py = self.previous_hand
            cx, cy = current_hand

            # 记录手的移动轨迹
            self.points.append([cx, cy])
            distance = math.hypot(cx - px, cy - py)
            self.lengths.append(distance)
            self.current_length += distance
            self.previous_hand = cx, cy

            # 长度缩减，使蛇不会无限制地变长
            if self.current_length > self.allowed_length:
                for i, length in enumerate(self.lengths):
                    self.current_length -= length
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.current_length < self.allowed_length:
                        break

            # 检查是否吃到食物
            rx, ry = self.food_point
            if abs(cx - rx) < self.w_food // 2 and abs(cy - ry) < self.h_food // 2:
                # 吃到食物，更新食物位置和分数
                self.random_food_location()
                self.allowed_length += 50
                self.score += 1
                print(self.score)

            # 画蛇
            self.draw_snake(img_main)

            # 画食物
            rx, ry = self.food_point
            img_main = cvzone.overlayPNG(img_main, self.img_food, (rx - self.w_food // 2, ry - self.h_food // 2))

            # 画分数
            cvzone.putTextRect(img_main, f'Score: {self.score}', (10, 50), scale=2, thickness=2, offset=10)

            # 检测是否碰到自己
            if self.detect_collision:
                self.check_collision(img_main, cx, cy)

        return img_main


# 初始化游戏对象
game = SnakeGameClass("Donut.png", "SnakeHead.png")

# 摄像头初始化
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# 手部检测器初始化
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    # 读取摄像头帧
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 获取手的位置
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        # 获取食指和拇指的位置
        index_finger_tip = hands[0]['lmList'][8]  # 食指尖端
        thumb_tip = hands[0]['lmList'][4]  # 拇指尖端

        # 计算食指和拇指的距离
        distance = ((index_finger_tip[0] - thumb_tip[0]) ** 2 + (index_finger_tip[1] - thumb_tip[1]) ** 2) ** 0.5

        # 如果食指和拇指的距离小于某个阈值，认为食指和拇指捏合
        if game.game_over and distance < 30:
            print('Start Game')
            game.game_over = False
            game.score = 0  # 重置得分
            game.reset_game()  # 重置游戏状态

    # 更新游戏
    img = game.update(img, game.previous_hand)

    # 显示图像
    cv2.imshow("Image", img)

    # 检测键盘输入
    key = cv2.waitKey(1)
    if key == ord('r'):
        game.game_over = False
        game.score = 0  # 重置得分
        game.reset_game()  # 重置游戏状态
    elif key == ord('q'):
        game.detect_collision = not game.detect_collision
