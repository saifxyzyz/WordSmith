# WordSmith: Your Professional Correspondent  
## Create Virtual environment  
```python -m venv my_hf_env```

## Activate Virtual environment  
On macOS/Linux  
```source my_hf_env/bin/activate```  

On Windows  
```my_hf_env\Scripts\activate```

After activation, your command line prompt will change to show the environment's name, like (my_hf_env)  

## install dependencies:
```pip install -r requirements.txt```  
### Note: If you have a GPU, read the commented lines.  
## Choose your dataset:
I have provided a few datasets to train the model on, but you can outsource a dataset as well. Make sure to replace the dataset value in train.py.  

## Train the model:  
Once you have decided what you want to fine-tune the model on, and made changes in the train.py script accordingly,  
Run:  
```train.py```   
P.S. This is going to take a good while  

## Once the model is done training
You can interact with it by running:  
```inference.py```


