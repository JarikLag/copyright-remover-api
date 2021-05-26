import hashlib
import io
import json
import os
import time

import requests
from PIL import Image, ImageOps
from selenium import webdriver


DRIVER_PATH = 'chromedriver.exe'
SEARCH_ENGINES = ['GOOGLE', 'YANDEX']
QUERIES = queries = ['sculpture',
                     'sculpture in museum']
MAX_LINKS = [5000, 5000]
DEFAULT_RESIZE = (1024, 1024)
YANDEX_ITER_THRESHOLD = 60


def fetch_image_urls_google(query: str, max_links_to_fetch: int, wd: webdriver, sleep_between_interactions: int = 2.5):
    search_url = f'https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={query}&oq={query}&gs_l=img'
    wd.get(search_url)

    last_height = wd.execute_script('return document.body.scrollHeight')
    while True:
        wd.execute_script('window.scrollTo(0,document.body.scrollHeight)')
        time.sleep(sleep_between_interactions)
        new_height = wd.execute_script('return document.body.scrollHeight')
        try:
            wd.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[4]/div[2]/input').click()
            time.sleep(sleep_between_interactions)
        except:
            pass
        if new_height == last_height:
            break
        last_height = new_height

    image_urls = set()
    thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
    number_results = len(thumbnail_results)
    print(f"Found: {number_results} search results. Extracting links.")

    limit = max_links_to_fetch if number_results > max_links_to_fetch else number_results

    for idx, img in enumerate(thumbnail_results[:limit]):
        try:
            img.click()
            time.sleep(sleep_between_interactions)
            actual_image = wd.find_element_by_xpath(
                '//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div/div[2]/a/img'
            )
            if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                url = actual_image.get_attribute('src')
                image_urls.add(url)
                print(f'{idx}/{limit}: {url}')
        except:
            continue

    return image_urls


def fetch_urls(wd, max_links_to_fetch, urls):
    results = wd.find_elements_by_css_selector("div.serp-item")
    number_results = len(results)
    links_remain = abs(max_links_to_fetch - len(urls))
    limit = min(links_remain, number_results)
    start_idx = len(urls)
    for idx, img in enumerate(results[:limit]):
        item = json.loads(img.get_attribute("data-bem"))
        try:
            url = item["serp-item"]["preview"][0]["url"]
            urls.add(url)
            print(f'{idx + start_idx}/{max_links_to_fetch}: {url}')
        except:
            print(f'Error while retrieving URL for image #{idx}')
            continue


def fetch_image_urls_yandex(query: str, max_links_to_fetch: int, wd: webdriver, sleep_between_interactions: int = 2):
    search_url = f'https://yandex.ru/images/search?text={query}'
    wd.get(search_url)

    image_urls = set()

    max_height = wd.execute_script('return document.body.scrollHeight')
    iters = 1
    cur_height = 0
    while cur_height < max_height:
        wd.execute_script(f'window.scrollTo(0,{cur_height})')
        time.sleep(sleep_between_interactions)
        cur_height += 1000
        if iters % YANDEX_ITER_THRESHOLD == 0:
            fetch_urls(wd, max_links_to_fetch, image_urls)
        iters += 1

    fetch_urls(wd, max_links_to_fetch, image_urls)
    print(f"Found: {len(image_urls)} search links. Downloading.")
    return image_urls


def download_images(urls, path_to_save, resize=False):
    os.makedirs(path_to_save, exist_ok=True)
    for idx, url in enumerate(urls):
        print(f'Processing url #{idx}')
        try:
            image_content = requests.get(url, timeout=(10, 10)).content
        except Exception as e:
            print(f'ERROR - Could not download {url} - {e}')

        try:
            image_file = io.BytesIO(image_content)
            image = Image.open(image_file).convert('RGB')
            if resize:
                image.thumbnail(DEFAULT_RESIZE, Image.ANTIALIAS)
            file_path = os.path.join(path_to_save, hashlib.sha1(image_content).hexdigest()[:15] + '.jpg')
            with open(file_path, 'wb') as f:
                image.save(f, "JPEG", quality=75)
            print(f"SUCCESS - saved {url} - as {file_path}")
        except Exception as e:
            print(f"ERROR - Could not save {url} - {e}")


def main():
    wd = webdriver.Chrome(executable_path=DRIVER_PATH)
    for engine in SEARCH_ENGINES:
        for idx, query in enumerate(QUERIES):
            if engine == 'GOOGLE':
                urls = fetch_image_urls_google(query, MAX_LINKS[idx], wd)
            elif engine == 'YANDEX':
                urls = fetch_image_urls_yandex(query, MAX_LINKS[idx], wd)
            else:
                print(f'Unknown search engine: {engine}')
                return
            path_to_save = f'downloads/{engine}/{query}'
            download_images(urls, path_to_save, resize=True)


if __name__ == '__main__':
    main()
