import requests
from bs4 import BeautifulSoup
import re
import os
import subprocess
import signal
import random
from . import proxy

def get_content(url):
    try:
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:45.0) Gecko/20100101 Firefox/45.0'
        proxy_ip = proxy.get_proxy()
        proxies = {'http': 'http://{}'.format(proxy_ip), 'https': 'https://{}'.format(proxy_ip)}
        response = requests.get(url, headers={'User-Agent': user_agent})#, proxies=proxies)
        response.raise_for_status()  # 如果返回的状态码不是200， 则抛出异常;
        response.encoding = response.apparent_encoding  # 判断网页的编码格式， 便于respons.text知道如何解码;
    except Exception as e:
        raise e
    else:
        #print(response.url)
        #print("爬取成功!")
        return response.content

def parser_content(htmlContent):
    # 实例化soup对象， 便于处理；
    soup = BeautifulSoup(htmlContent, 'html.parser')
    # 提取页面的头部信息， 解决乱码问题

    # 提取需要的内容;
    sound_urls = soup.find_all('a', class_="title")
    urls = []
    for i in sound_urls:
        urls.append(i.get('href'))
    return urls


def download_sound(text, save_path, timeout=60.0):

    url = "https://freesound.org/search/?q="+text
    content = get_content(url)
    if not content:
        return []
    urls = parser_content(content)
    if not urls:
        return []

    record = []

    cnt = 0 # 成功下载的文件数
    for u in urls:
        if not cnt < 5:
            break
        file_url = 'https://freesound.org' + u
        #print(file_url)

        # 判断文件大小，对于过大的文件直接放弃
        p = subprocess.Popen('you-get --info %s' % file_url, stdout=subprocess.PIPE, shell=True)
        info_str = p.communicate()[0].decode('utf-8')
        pattern = re.compile(r'(\d+\.\d*)[\s]MiB')
        size = float(pattern.findall(info_str)[0])
        #print("size=%.2f Mib"%size)
        if size > 3.0:
            continue

        cmd = 'you-get --skip-existing-file-size-check -o %s -O %s  %s' % (save_path.replace("'", "\\'"),
            str(cnt), file_url)
        if random.randint(0, 5) < 5:
            cmd += ' -x {}'.format(proxy.get_proxy())
        cmd += ' > tmp.txt'
        #print(cmd)

        p = subprocess.Popen(cmd, start_new_session=True, shell=True)
        try:
            (msg, errs) = p.communicate(timeout=timeout)
            ret_code = p.poll()
            cnt += 1
        except subprocess.TimeoutExpired: # 若下载超时 (文件过大)，则终止下载进程，并清除临时文件
            p.kill()
            p.terminate()
            os.killpg(p.pid, signal.SIGTERM)
            subprocess.run('rm '+save_path.replace("'", "\\'")+'/*.download', shell=True)
            continue
    for file in os.listdir(save_path):
        record.append(os.path.splitext(file)[1][1:])
    return record

if __name__ == '__main__':
    download_sound('dog','Sounds/dog')
