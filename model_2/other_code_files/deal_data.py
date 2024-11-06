#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os.path
from bs4 import BeautifulSoup
import pandas as pd
import re

def deal_test_data(csv_path, html_dir, image_dir, model_dir):

    # 将新闻内容进行提取和组合
    # 获取csv文件内容
    csv_content = pd.read_csv(csv_path)
    # id,Ofiicial Account Name,Title                      ,News Url,Image Url,Report Content,label
    # 0, 朝阳实拍               , 范冰冰出轨被偷拍？这才是真相！, http    ,  http   ,发布误导信息    ,1  (1 是假新闻，0是真新闻)
    i = 0
    data_length = len(csv_content)
    while i < data_length:
        # news_id = single_news['id']
        # official_account_name = single_news['Ofiicial Account Name']
        news_id = i
        # news_title = csv_content.loc[i, 'Title']
        # report_content = csv_content.loc[i, 'Report Content']

        # 根据ID获取image和html
        image_file_name = str(news_id) + '.png'
        html_file_name = str(news_id) + '.html'

        image_path = os.path.join(image_dir, image_file_name)
        html_path = os.path.join(html_dir, html_file_name)

        # 获取image和html中的文本内容
        # 打开HTML文件
        with open(html_path) as file:
            html = file.read()

        # 创建BeautifulSoup对象
        soup = BeautifulSoup(html, 'html.parser')

        # 提取所有文本
        html_content_text = soup.get_text()

        # 清洗数据 去除空格
        html_content_text = html_content_text.replace('\n', '')
        html_content_text = html_content_text.replace('\t', '')
        html_content_text = html_content_text.replace(u'\xa0', '')
        html_content_text = html_content_text.replace(' ', '')
        html_content_text = html_content_text.replace('"', '')

        html_content_text_dealed = html_content_text.strip()
        # total_dealed_text = str(news_title) + str(html_content_text_dealed) + str(report_content)
        total_dealed_text = str(html_content_text_dealed)

        # total_dealed_text_list.append(total_dealed_text)
        # 保存处理后的数据
        text_label_dict = {'text': total_dealed_text}
        dealed_file = pd.DataFrame(text_label_dict, index=[i])

        dealed_file.to_csv(model_dir + '/other_code_files/dealed_test_data.csv', mode='a', index=True, header=False)
        i += 1


def deal_train_data(to_train_dir):

    # 记录真实新闻官方账号名称
    real_official_name = []

    train_csv_path = os.path.join(to_train_dir, "train.csv")
    train_html_dir = os.path.join(to_train_dir, "html")
    train_image_dir = os.path.join(to_train_dir, "image")

    csv_content = pd.read_csv(train_csv_path)
    # id,Ofiicial Account Name,Title                      ,News Url,Image Url,Report Content,label
    # 0, 朝阳实拍               , 范冰冰出轨被偷拍？这才是真相！, http    ,  http   ,发布误导信息    ,1  (1 是假新闻，0是真新闻)
    i = 0
    for single_news in csv_content.values:
        news_id = single_news[0]
        official_account_name = single_news[1]
        news_title = single_news[2]
        report_content = single_news[5]
        news_label = single_news[6]

        # 根据ID获取image和html
        image_file_name = str(news_id) + '.png'
        html_file_name = str(news_id) + '.html'

        image_path = os.path.join(train_image_dir, image_file_name)
        html_path = os.path.join(train_html_dir, html_file_name)

        # 获取image和html中的文本内容
        # 打开HTML文件
        with open(html_path) as file:
            html = file.read()

        # 创建BeautifulSoup对象
        soup = BeautifulSoup(html, 'html.parser')

        # 提取所有文本
        html_content_text = soup.get_text()

        # 清洗数据 去除空格
        html_content_text = html_content_text.replace('\n', '')
        html_content_text = html_content_text.replace('\t', '')
        html_content_text = html_content_text.replace(u'\xa0', '')
        html_content_text = html_content_text.replace(' ', '')
        html_content_text = html_content_text.replace('"', '')
        html_content_text_dealed = html_content_text.strip()

        total_dealed_text = str(news_title) + str(html_content_text_dealed) + str(report_content)

        if official_account_name not in real_official_name and news_label == 0:
            real_official_name.append(official_account_name)

        text_label_dict = {'text': total_dealed_text, 'label': news_label}
        dealed_file = pd.DataFrame(text_label_dict, index=[i])
        dealed_file.to_csv('./dealed_train_data.csv', mode='a', index=False, header=False)
        i += 1

if __name__ == '__main__':
    to_train_dir = '/Users/mazhenyu/code/study/computer_challenge/train'
    deal_train_data(to_train_dir)

