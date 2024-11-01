import argparse
import json
import os
import random
import scrapetube

from time import sleep
from tqdm import tqdm

from selenium import webdriver

from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def get_data(driver, wait_time, videoid):
    title = None
    data = []

    try:
        driver.get('https://www.youtube.com/watch?v=%s&vq=small' % videoid)
        wait = WebDriverWait(driver, wait_time)
    except Exception as e:
        return {'status': 'driver_error', 'message': str(e)}

    # get title
    try:
        title_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="title"]/h1/yt-formatted-string')))
        title = title_element.text
    except Exception as e:
        return {'status': 'title_error', 'message': str(e)}
        
    # click expand
    try:
        expand_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="expand"]')))
        expand_button.click()
    except Exception as e:
        return {'status': 'expand_error', 'message': str(e)}
    
    # load transcript
    try:
        transcript_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[contains(@aria-label, "Show transcript")]')))
        driver.execute_script('arguments[0].scrollIntoView();', transcript_button) # scroll
        transcript_button.click()
    except Exception as e:
        return {'status': 'load_transcript_error', 'message': str(e)}

    # get transcript
    try:
        container = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="segments-container"]')))
        children = container.find_elements(By.XPATH, './*')
        for _, child in enumerate(children):
            tag_name = child.tag_name

            if tag_name == 'ytd-transcript-section-header-renderer':
                data.append({'header': child.text})
            elif tag_name == 'ytd-transcript-segment-renderer':
                data.append({'segment': child.text})
        return {'status': 'success', 'title': title, 'data': data}
    except Exception as e:
        return {'status': 'scrape_transcript_error', 'message': str(e)}
            
    

def main():
    parser = argparse.ArgumentParser(description='Scrape Transcripts for a Youtube Channel')
    
    parser.add_argument('--channel_id', type=str, default='UC2D2CMWXMOVWx7giW1n3LIg',
                        help='channel id of a youtube channel')
    parser.add_argument('--wait_time', type=int, default=10)
    parser.add_argument('--sleep_time', type=int, default=10)
    parser.add_argument('--output_file_path', type=str, default='data.json')
    
    args = parser.parse_args()
    
    # get video ids from the youtube channel
    videos = scrapetube.get_channel(args.channel_id)
    video_ids = [video['videoId'] for video in videos]
    print('> {} videos found!'.format(len(video_ids)))
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # exclude already processed video_ids
    if os.path.exists(args.output_file_path):
        print('> {} already exists!'.format(args.output_file_path))
        with open(args.output_file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
            processed_video_ids = [item['video_id'] for item in data]
            new_video_ids = []
            for id in video_ids:
                if id not in processed_video_ids:
                    new_video_ids.append(id)
        print('> {}/{} left to process'.format(len(new_video_ids), len(video_ids)))
        video_ids = new_video_ids
    else:   
        with open(args.output_file_path, 'w', encoding='utf-8') as f:
            f.write('')
    
    for video_id in tqdm(video_ids, total=len(video_ids)):
        result = get_data(driver, args.wait_time, video_id)

        if result['status'] == 'success':
            output_data = {'video_id': video_id, 'title': result['title'], 'transcript': result['data']}
            with open(args.output_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
        else:
            print('Error: {}\n'.format(video_id), result['status'])
            
        sleep(random.uniform(args.sleep_time-5, args.sleep_time+5))
        
    driver.quit()
    
    
if __name__ == '__main__':
    main()