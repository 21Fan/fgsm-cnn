# coding:utf-8
# name:captcha_gen.py

import random
import numpy as np
from captcha.image import ImageCaptcha


NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOW_CASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
UP_CASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
           'V', 'W', 'X', 'Y', 'Z']

CAPTCHA_LIST = NUMBER
CAPTCHA_LEN = 4         # 验证码长度
CAPTCHA_HEIGHT = 60     # 验证码高度
CAPTCHA_WIDTH = 160     # 验证码宽度
CAPTCHA_CHANNEL=1

def random_captcha_text(char_set=CAPTCHA_LIST, captcha_size=CAPTCHA_LEN):
    """
    随机生成定长字符串
    :param char_set: 备选字符串列表
    :param captcha_size: 字符串长度
    :return: 字符串
    """
    captcha_text = [random.choice(char_set) for _ in range(captcha_size)]
    return ''.join(captcha_text)


def gen_captcha_text_and_image(height=CAPTCHA_HEIGHT,width=CAPTCHA_WIDTH,channel=CAPTCHA_CHANNEL, save=None):
    """
    生成随机验证码
    :param width: 验证码图片宽度
    :param height: 验证码图片高度
    :param save: 是否保存（None）
    :return: 验证码字符串，验证码图像np数组
    """
    image = ImageCaptcha(width=width, height=height)
    # 验证码文本
    captcha_text = random_captcha_text()
    if channel==3:
        captcha = image.generate_image(captcha_text)#生成三通道
    else:
        captcha = image.generate_image(captcha_text).convert('L')
    # 保存
    if save=='train':
        image.write(captcha_text, './img/' + captcha_text + '.jpg')#修改成这种格式./img/
    elif save =='adv':
        image.write(captcha_text, './before_adv/' + captcha_text + '.jpg')  # 修改成这种格式./before_adv/

    captcha = np.array(captcha.getdata())
    captcha = np.reshape(captcha, [height, width, channel]) / 255.0#降成0到1
    # np.set_printoptions(threshold=9999)  # 打印全部
    # print(captcha)
    return captcha, captcha_text


if __name__ == '__main__':
    im, t = gen_captcha_text_and_image(save='train')

    print(im.shape,t)      # (60, 160, 3)
