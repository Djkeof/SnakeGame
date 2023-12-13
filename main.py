from asyncio import current_task
import math
import random
from threading import currentThread
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)


class SnakeGameClass:

    def __init__(self,pathFood,pathSnakeHead):
        self.detect_collision = True  # 默认开启碰撞检测

        self.points = [] #蛇的所有节点
        self.lengths = [] #每个节点间距离
        self.currentLength = 0 #蛇当前长度
        self.allowedLength = 150 #蛇最大长度
        self.previousHand = 0,0 #上一个节点

        self.imgSnakeHead = cv2.imread(pathSnakeHead, cv2.IMREAD_UNCHANGED)  # 读取蛇头图片
        print(f"Image shape: {self.imgSnakeHead.shape}")
        print(f"Number of channels: {self.imgSnakeHead.shape[2]}")
        self.imgSnakeHead = cv2.cvtColor(self.imgSnakeHead, cv2.COLOR_BGRA2BGR)
        self.imgSnakeHead = cv2.resize(self.imgSnakeHead, (50, 50))  # 调整蛇头图片的大小
        self.hSnakeHead, self.wSnakeHead, _ = self.imgSnakeHead.shape  # 获取蛇头图片的高度和宽度
        

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0,0
        self.randomFoodLocation()

        self.move_threshold = 10  # 移动阈值

        self.frame_counter = 0  # 帧计数器
        self.frame_interval = 5  # 检测间隔

        self.scoer = 0
        self.gameOver = False

        self.donuts = []  # 存储当前存在的 Donut
        self.lastDonutTime = time.time()  # 记录上一个 Donut 出现的时间
        self.donutInterval = 2  # 每隔一秒生成一个 Donut
        self.donutDuration = 15  # 每个 Donut 存在两秒

        self.gameStarted = False  # 游戏是否已开始
        self.pinch_threshold = 10  # 设定捏合手势的距离阈值
        self.pinch_detected = False  # 用于检测捏合手势的标志
    def resetGame(self):
        self.points = []  # 蛇的所有节点
        self.lengths = []  # 每个节点间距离
        self.currentLength = 0  # 蛇当前长度
        self.allowedLength = 150  # 蛇最大长度
        self.previousHand = 0, 0  # 上一个节点
        self.randomFoodLocation()

    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)
        

    def update(self,imgMain,currentHand):
        

        if self.gameOver:
            cvzone.putTextRect(imgMain,"Game Over",(300,400),scale=5,thickness=10,offset=20)
            cvzone.putTextRect(imgMain,f'Score: {self.scoer}',(300,550),scale=5,thickness=10,offset=20)
        else:
            
            
            px,py = self.previousHand
            cx,cy = currentHand

            self.points.append([cx,cy])
            distance = math.hypot(cx-px,cy-py)
            self.lengths.append(distance)
            self.currentLength += distance
            self.previousHand = cx,cy

            

            #长度缩减
            if self.currentLength > self.allowedLength:
                for i,length in enumerate(self.lengths):
                    self.currentLength -= length
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.currentLength < self.allowedLength:
                        break
            
            
            
            # 检查是否吃到 Donut
            rx,ry = self.foodPoint
            if abs(cx-rx)<self.wFood//2 and abs(cy-ry)<self.hFood//2:         
                self.randomFoodLocation()
                self.allowedLength += 50
                self.scoer += 1
                print(self.scoer)

            

            #画蛇
            # if self.points:
            #     for i,point in enumerate(self.points):
            #         if i != 0:
            #             cv2.line(imgMain,self.points[i-1],self.points[i],(0,0,255),20)
            #     cv2.circle(imgMain,self.points[-1] , 20, (200, 0, 200), cv2.FILLED)
            #     cx, cy = self.points[-1]
            #     imgMain = cvzone.overlayPNG(imgMain, self.imgSnakeHead, 
            #                                 (cx-self.wSnakeHead//2, cy-self.hSnakeHead//2))
            if self.points:
                body_pts = np.array(self.points[:-2], np.int32)
                body_pts = body_pts.reshape((-1, 1, 2))
                cv2.polylines(imgMain, [body_pts], False, (0, 0, 255), 20)
                cx, cy = self.points[-1]
                cv2.circle(imgMain, (cx, cy), 20, (200, 0, 200), cv2.FILLED)
                # print(f"Image array size: {self.imgSnakeHead.shape}")
                

                cx, cy = self.points[-1]  # 获取蛇头的坐标

                # 计算蛇头图片的位置
                start_x = max(0, cx - self.wSnakeHead // 2)  # 保证起始横坐标不小于0
                end_x = min(imgMain.shape[1], start_x + self.imgSnakeHead.shape[1])  # 保证结束横坐标不超过图像宽度
                start_y = max(0, cy - self.hSnakeHead // 2)  # 保证起始纵坐标不小于0
                end_y = min(imgMain.shape[0], start_y + self.imgSnakeHead.shape[0])  # 保证结束纵坐标不超过图像高度

                # 将蛇头图片覆盖到 imgMain 上
                imgMain[start_y:end_y, start_x:end_x] = self.imgSnakeHead[:end_y-start_y, :end_x-start_x]

                # imgMain = cvzone.overlayPNG(imgMain, self.imgSnakeHead, (cx - self.wSnakeHead // 2, cy - self.hSnakeHead // 2))

                
            #画食物
            rx,ry = self.foodPoint
            imgMain = cvzone.overlayPNG(imgMain,self.imgFood,(rx-self.wFood//2,ry-self.hFood//2))
            
            #画分数
            cvzone.putTextRect(imgMain,f'Score: {self.scoer}',(10,50),scale=2,thickness=2,offset=10)

            #吃自己
            

            if self.detect_collision:
            # 忽略蛇身的最后几个节点
                pts = np.array(self.points[:-4],np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(imgMain,[pts],False,(0,200,0),3)
                minDist = cv2.pointPolygonTest(pts,(cx,cy),True)
            
            # for i, point in enumerate(self.points[:-2]):
            #     px, py = point
            #     if i != 0 and abs(cx - px) < self.wSnakeHead and abs(cy - py) < self.hSnakeHead:
            #         print("Game Over - 头部与身体接触")
            #         self.gameOver = True
            #         self.resetGame()
            #         break
                

                if -1 <= minDist <= 1:
                    print("Game Over")
                    self.gameOver = True
                    self.points = [] #蛇的所有节点
                    self.lengths = [] #每个节点间距离
                    self.currentLength = 0 #蛇当前长度
                    self.allowedLength = 150 #蛇最大长度
                    self.previousHand = 0,0 #上一个节点
                    self.randomFoodLocation()

            

        return imgMain

game = SnakeGameClass("Donut.png", "SnakeHead.png")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]
        img = game.update(img, pointIndex)

        # 获取食指和拇指的位置
        index_finger_tip = hands[0]['lmList'][8]  # 食指尖端
        thumb_tip = hands[0]['lmList'][4]  # 拇指尖端

        # 计算食指和拇指的距离
        distance = ((index_finger_tip[0] - thumb_tip[0]) ** 2 + (index_finger_tip[1] - thumb_tip[1]) ** 2) ** 0.5

        # 如果食指和拇指的距离小于某个阈值，认为食指和拇指捏合
        if game.gameOver and distance < 30:
            print('Start Game')
            game.gameOver = False
            game.scoer = 0  # 重置得分
            game.gameStarted = False  # 重置游戏状态

        
            
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        game.gameOver = False
        game.scoer = 0  # 重置得分
        game.gameStarted = False  # 重置游戏状态
    elif key == ord('q'):
        game.detect_collision = not game.detect_collision
    