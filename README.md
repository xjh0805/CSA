# CSA
The codes for our paper "CSA: Cross-scale alignment with adaptive semantic aggregation and filter for image-text retrieval"

The repository includes all the scripts, configurations, and detailed instructions on how to execute the code and reproduce our results. Once the paper is accepted, we will open-source our code.


## Requirements
- Python 3.7.11
- PyTorch 1.7.1
- NumPy 1.21.5
- Punkt Sentence Tokenizer:
   ```
   import nltk
   nltk.download()
   >d punkt
## Data pre-processing (Optional) 
   The image features of Flickr30K and MS-COCO are available in numpy array format, which can be used for training directly. However, if you wish to test on another dataset, you will need to start from scratch:

   
   1、Use the and the bottom-up attention model to extract features of image regions. The output file format will be a tsv, where the columns are ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features'].  `bottom-up-attention/tools/generate_tsv.py`

   
   2、Use to convert the above output to a numpy `array.util/convert_data.py`

   
## Download data and vocab
We follow SCAN to obtain image features and vocabularies, which can be downloaded by using:
```
https://www.kaggle.com/datasets/kuanghueilee/scan-features
```


Another download link is available below：
```
https://drive.google.com/drive/u/0/folders/1os1Kr7HeTbh8FajBNegW8rjJf6GIhFqC
```

## Training
```
python train.py --data_path "$DATA_PATH" --data_name f30k_precomp --vocab_path "$VOCAB_PATH" --logger_name runs/log --logg_path runs/runX/logs --model_name "$MODEL_PATH"
```
