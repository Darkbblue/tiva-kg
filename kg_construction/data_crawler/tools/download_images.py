# -*- coding: utf-8 -*-

import os
import re
import requests
from . import proxy


def download_image(text, save_path, limits=5):
    url = "https://www.google.com/search?q=" + text.replace(" ", "+") + "&source=lnms&tbm=isch"
    # print(url)
    headers = {
        #'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:86.0) Gecko/20100101 Firefox/86.0',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
        'Connection': 'close'
    }
    proxy_ip = proxy.get_proxy()
    proxies = {'http': 'http://{}'.format(proxy_ip), 'https': 'https://{}'.format(proxy_ip)}
    try:
        response = requests.get(url, headers=headers, proxies=proxies, timeout=10)
    except Exception as e:
        raise e
    # with open("html.txt", "w", encoding='utf-8') as f:
    #     f.write(response.text)
    image_urls = re.findall(r'\["(https://[^ ,]*?\.jpg)",', response.text)
    # print(len(image_urls))
    #
    # print("start downloading images!")

    record = []

    cnt = 0
    for i, url in enumerate(image_urls):
        if cnt >= limits:
            break
        #print(url)
        filename = os.path.join(save_path, str(cnt) + ".jpg")
        #print(filename)
        try:
            r = requests.get(url, headers=headers, proxies=proxies, stream=True, timeout=45)
            r.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(r.content)
            cnt += 1
        except Exception as e:
            # print(e)
            continue
        record.append('jpg')
    # print("finish downloading images!")
    return record
