import os.path
from bs4 import BeautifulSoup
import pandas as pd


def deal_news_text_data(csv_path, html_dir, image_dir, pattern_name):

    # 将新闻内容进行提取和组合
    # 获取csv文件内容
    csv_content = pd.read_csv(csv_path)
    # id,Ofiicial Account Name,Title                      ,News Url,Image Url,Report Content,label
    # 0, 朝阳实拍               , 范冰冰出轨被偷拍？这才是真相！, http    ,  http   ,发布误导信息    ,1  (1 是假新闻，0是真新闻)
    for single_news in csv_content.values:
        news_id = single_news[0]
        official_account_name = single_news[1]
        news_title = single_news[2]
        report_content = single_news[5]
        news_label = single_news[6]

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

        # 清洗数据 去除空格 判断结尾停止位置
        html_content_text = html_content_text.replace("\n", "")
        html_content_text_dealed = html_content_text.strip()
        print(html_content_text_dealed)
    return 0

if __name__ == '__main__':
    to_pred_dir = "/Users/mazhenyu/code/study/computer_challenge/train"
    to_train_dir = os.path.abspath(to_pred_dir)
    train_csv_path = os.path.join(to_pred_dir, "train.csv")
    train_html_dir = os.path.join(to_pred_dir, "html")
    train_image_dir = os.path.join(to_pred_dir, "image")
    deal_news_text_data(train_csv_path, train_html_dir, train_image_dir, "train")