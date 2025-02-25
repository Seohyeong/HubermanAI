import json
import re
import os

import torch
from transformers import pipeline

raw_data_path = 'data/raw_data.json'
train_data_path = 'data/train.json' # formatted raw_data
qna_test_data_path = 'data/qna_test.json' # test dataset created from the qna sessions
syn_test_data_path = 'data/syn_test.json' # test dataset created synthetically 
        
def create_train_data(raw_json):
    docs = []
    qna_docs = []
    for item in raw_json:
        video_id = item['video_id']
        video_title = item['title']
        
        video_header = ''
        segment_idx = 0
        chunks = []
        for chunk in item['transcript']:
            chunk_type = list(chunk.keys())[0] # 'header' or 'segment'
            if chunk_type == 'header':
                if chunks and video_header:
                    context = ' '.join([item['text'] for item in chunks])
                    time_start = chunks[0]['timestamp']
                    time_end = chunks[-1]['timestamp']
                    
                    doc = {'doc_id': video_id + '_' + str(segment_idx),
                            'video_id': video_id, 
                            'video_title': video_title, 
                            'video_header': video_header,
                            'segment_idx': str(segment_idx),
                            'time_start': time_start,
                            'time_end': time_end,
                            'context': context}
                    
                    docs.append(doc)
                    
                    # extract qna sessions
                    if 'LIVE EVENT Q&A' in video_title:
                        qna_docs.append(doc)
                        
                    video_header = ''
                    segment_idx += 1
                    chunks = []
                video_header = list(chunk.values())[0]
            else:
                try:
                    timestamp, text = list(chunk.values())[0].split('\n', 1)
                except:
                    continue # {'segment': '0:07'}
                text = re.sub(r'\[.*?\] ', '', text)
                chunks.append({'timestamp': timestamp, 'text': text})  
    return docs, qna_docs
        

with open(raw_data_path, 'r', encoding='utf-8') as f:
        raw_json = [json.loads(line) for line in f]

train_docs, qna_test_docs = create_train_data(raw_json)

# train dataset
if not os.path.exists(train_data_path):
    with open(train_data_path, 'a', encoding='utf-8') as f:
        for doc in train_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')   
        

# create qna test dataset
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

qna_test_docs_w_questions = []
for doc in qna_test_docs:
    if 'sponsor' not in doc['video_header'].lower():
        
        user_msg = 'header: ' + doc['video_header'] + '\n' + 'context: ' + doc['context']
        messages = [
            {"role": "system", "content": "You are an AI agent to help creating Question and Answering dataset. Given the context and the header of the context, generate a question that can be answered with the context."},
            {"role": "user", "content": user_msg},
        ]
        outputs = pipe(
            messages,
            max_new_tokens=256,
        )
        
        print(outputs[0]["generated_text"][-1])